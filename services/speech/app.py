"""Internal FastAPI boundary for the canonical batch speech service."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from mtbank_ai.api.body_limits import BodyLimitMiddleware
from mtbank_ai.api.error_handlers import install_error_handlers
from mtbank_ai.api.schemas import HealthResponse
from mtbank_ai.domain.errors import DomainError, ErrorCode, ErrorResponse, build_error_response
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.speech.contracts import SpeechFile, SpeechMetadata, SpeechTranscriptionResponse
from mtbank_ai.speech.roles import RoleResolutionRequiredError
from mtbank_ai.speech.streaming import (
    StreamingAdapterUnavailable,
    StreamingSpeechSession,
    StreamingUpdate,
    parse_streaming_binary_frame,
    parse_streaming_end,
    parse_streaming_start,
    validate_stream_frame,
)
from services.speech.errors import (
    MediaTimeoutError,
    MediaValidationError,
    NoSpeechError,
    SpeechConfigurationError,
    SpeechDeadlineExceededError,
    SpeechOverloadedError,
    SpeechProviderError,
    UnsupportedMediaError,
)
from services.speech.runtime import LazySpeechRuntime, SpeechRuntimePort, StreamingRuntimePort, UnavailableSpeechRuntime
from services.speech.settings import (
    FasterWhisperSettings,
    SpeechAccessSettings,
    SpeechModelSettings,
    SpeechRuntimeSettings,
    SpeechSettings,
)

_LOGGER = logging.getLogger(__name__)
_MULTIPART_RESERVE_BYTES = 64 * 1024
_TRANSCRIBE_PATHS = frozenset({"/v1/transcribe", "/v1/transcribe/"})

_UNAUTHENTICATED_RESPONSE = {401: {"model": ErrorResponse, "description": "Bearer authentication is required"}}

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    **_UNAUTHENTICATED_RESPONSE,
    **{
        status: {"model": ErrorResponse, "description": description}
        for status, description in (
            (409, "Нужно подтвердить роли говорящих"),
            (413, "Превышен допустимый размер"),
            (415, "Неподдерживаемый media type"),
            (422, "Некорректное аудио или metadata"),
            (429, "Speech queue заполнена"),
            (502, "ASR provider завершился с ошибкой"),
            (503, "Локальные model artifacts не готовы"),
            (504, "Истёк deadline обработки"),
        )
    },
}


def create_app(
    settings: SpeechSettings | None = None,
    runtime: SpeechRuntimePort | None = None,
) -> FastAPI:
    resolved_settings, settings_failed = _resolve_settings(settings)
    resolved_runtime = runtime or _build_runtime(resolved_settings, settings_failed)
    resolved_streaming_runtime = _streaming_runtime(resolved_runtime)
    warmup_task: asyncio.Task[bool] | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal warmup_task
        del app
        if resolved_settings.runtime.device == "cuda" and isinstance(resolved_runtime, LazySpeechRuntime):
            # A failed model load does not kill liveness, but it permanently fails readiness.
            warmup_task = asyncio.create_task(_warmup_cuda_runtime(resolved_runtime))
        try:
            yield
        finally:
            if warmup_task is not None:
                # asyncio cannot force-cancel an active to_thread worker; warmup marks CUDA fatal instead.
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task
            result = resolved_runtime.close()
            if inspect.isawaitable(result):
                await result

    app = FastAPI(title="MTBank Canonical Speech Service", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.runtime = resolved_runtime
    app.add_middleware(
        BodyLimitMiddleware,
        max_json_bytes=resolved_settings.runtime.max_upload_bytes + _MULTIPART_RESERVE_BYTES,
        max_upload_bytes=resolved_settings.runtime.max_upload_bytes,
        multipart_reserve_bytes=_MULTIPART_RESERVE_BYTES,
        paths=_TRANSCRIBE_PATHS,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request_id = _request_id_from_header(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        if settings_failed and request.url.path != "/health/live":
            status, body = build_error_response(DomainError(ErrorCode.SERVICE_UNAVAILABLE), request_id)
            return JSONResponse(
                status_code=status,
                content=body.model_dump(mode="json"),
                headers={"X-Request-ID": str(request_id)},
            )
        if not _request_is_authorized(request, resolved_settings):
            status, body = build_error_response(DomainError(ErrorCode.UNAUTHENTICATED), request_id)
            return JSONResponse(
                status_code=status,
                content=body.model_dump(mode="json"),
                headers={"X-Request-ID": str(request_id), "WWW-Authenticate": "Bearer"},
            )
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", str(request_id))
        return response

    install_error_handlers(app)

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={
            **_UNAUTHENTICATED_RESPONSE,
            503: {"model": ErrorResponse, "description": "Speech artifacts are unavailable"},
        },
        tags=["health"],
    )
    async def ready() -> HealthResponse:
        if not await _runtime_is_ready(resolved_runtime, warmup_task):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
        return HealthResponse(status="ready")

    @app.get(
        "/v1/runtime",
        responses={
            **_UNAUTHENTICATED_RESPONSE,
            503: {"model": ErrorResponse, "description": "Speech runtime is unavailable"},
        },
        tags=["speech"],
    )
    async def runtime_attestation() -> dict[str, object]:
        if not await _runtime_is_ready(resolved_runtime, warmup_task):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
        return _runtime_attestation(resolved_settings, resolved_runtime)

    @app.post(
        "/v1/transcribe",
        response_model=SpeechTranscriptionResponse,
        responses=_ERROR_RESPONSES,
        tags=["speech"],
    )
    async def transcribe(request: Request) -> SpeechTranscriptionResponse:
        if await _cuda_warmup_blocks_transcription(resolved_settings, resolved_runtime, warmup_task):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
        source = await _parse_source(request, resolved_settings)
        if not await _runtime_is_ready(resolved_runtime, warmup_task):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
        try:
            return await resolved_runtime.transcribe(source)
        except UnsupportedMediaError as error:
            raise DomainError(ErrorCode.UNSUPPORTED_MEDIA) from error
        except (MediaTimeoutError, SpeechDeadlineExceededError) as error:
            raise DomainError(ErrorCode.DEADLINE_EXCEEDED) from error
        except MediaValidationError as error:
            raise DomainError(ErrorCode.INVALID_AUDIO) from error
        except NoSpeechError as error:
            raise DomainError(ErrorCode.NO_SPEECH) from error
        except RoleResolutionRequiredError as error:
            raise DomainError(ErrorCode.ROLE_RESOLUTION_REQUIRED) from error
        except SpeechOverloadedError as error:
            raise DomainError(ErrorCode.QUOTA_EXCEEDED) from error
        except SpeechConfigurationError as error:
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from error
        except SpeechProviderError as error:
            raise DomainError(ErrorCode.PROVIDER_FAILURE) from error

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        if settings_failed:
            await websocket.close(code=1013)
            return
        if not _websocket_is_authorized(websocket, resolved_settings):
            await websocket.close(code=1008)
            return
        if (
            not resolved_settings.streaming.enabled
            or resolved_streaming_runtime is None
            or not await resolved_runtime.ready()
        ):
            await websocket.close(code=1013)
            return
        session: StreamingSpeechSession | None = None
        bytes_received = 0
        sequence = 0
        deadline = asyncio.get_running_loop().time() + resolved_settings.streaming.max_duration_seconds
        try:
            await _within_stream_deadline(websocket.accept(), deadline)
            start_message = await _within_stream_deadline(websocket.receive_text(), deadline)
            start = parse_streaming_start(_json_object(start_message))
            session = await _within_stream_deadline(resolved_streaming_runtime.open_stream(start), deadline)
            assert session is not None
            await _within_stream_deadline(websocket.send_json({"type": "started", "sequence": 0}), deadline)
            while True:
                message = await _within_stream_deadline(websocket.receive(), deadline)
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    next_sequence, frame = parse_streaming_binary_frame(message["bytes"], sequence + 1)
                    if (
                        len(frame) > resolved_settings.streaming.max_frame_bytes
                        or bytes_received + len(frame) > resolved_settings.streaming.max_session_bytes
                    ):
                        await websocket.close(code=1009)
                        return
                    validate_stream_frame(start, frame, first_frame=sequence == 0)
                    bytes_received += len(frame)
                    sequence = next_sequence
                    updates = await _within_stream_deadline(
                        session.push(frame, sequence=sequence),
                        min(
                            deadline,
                            asyncio.get_running_loop().time() + resolved_settings.streaming.processing_timeout_seconds,
                        ),
                    )
                    await _send_stream_updates(
                        websocket,
                        updates,
                        deadline,
                        max_update_text_bytes=resolved_settings.streaming.max_update_text_bytes,
                    )
                    await _within_stream_deadline(websocket.send_json({"type": "ack", "sequence": sequence}), deadline)
                    continue
                if message.get("text") is not None:
                    payload = _json_object(message["text"])
                    parse_streaming_end(payload, sequence + 1)
                    sequence += 1
                    updates = await _within_stream_deadline(session.finish(), deadline)
                    await _send_stream_updates(
                        websocket,
                        updates,
                        deadline,
                        max_update_text_bytes=resolved_settings.streaming.max_update_text_bytes,
                    )
                    await _within_stream_deadline(
                        websocket.send_json({"type": "finished", "sequence": sequence}),
                        deadline,
                    )
                    return
                await websocket.close(code=1008)
                return
        except TimeoutError:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close(code=1013)
        except (
            SpeechConfigurationError,
            SpeechDeadlineExceededError,
            SpeechOverloadedError,
            SpeechProviderError,
            StreamingAdapterUnavailable,
        ):
            if websocket.client_state.name == "CONNECTED":
                await websocket.close(code=1013)
        except RuntimeError:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close(code=1013)
        except (ValueError, WebSocketDisconnect):
            if websocket.client_state.name == "CONNECTED":
                await websocket.close(code=1008)
        finally:
            if session is not None:
                await _close_stream_session(session)

    return app


async def _warmup_cuda_runtime(runtime: LazySpeechRuntime) -> bool:
    try:
        await runtime.warmup()
    except SpeechConfigurationError:
        _LOGGER.error('{"event":"speech_cuda_warmup_failed"}')
        return False
    return await runtime.ready()


async def _runtime_is_ready(runtime: SpeechRuntimePort, warmup_task: asyncio.Task[bool] | None) -> bool:
    if warmup_task is not None:
        if not warmup_task.done():
            return False
        try:
            if not warmup_task.result():
                return False
        except asyncio.CancelledError:
            return False
    return await runtime.ready()


async def _cuda_warmup_blocks_transcription(
    settings: SpeechSettings,
    runtime: SpeechRuntimePort,
    warmup_task: asyncio.Task[bool] | None,
) -> bool:
    return (
        settings.runtime.device == "cuda"
        and isinstance(runtime, LazySpeechRuntime)
        and not await _runtime_is_ready(runtime, warmup_task)
    )


async def _within_stream_deadline(awaitable: Any, deadline: float) -> Any:
    async with asyncio.timeout_at(deadline):
        return await awaitable


async def _send_stream_updates(
    websocket: WebSocket,
    updates: tuple[StreamingUpdate, ...],
    deadline: float,
    *,
    max_update_text_bytes: int,
) -> None:
    for update in updates:
        if len(update.text.encode("utf-8")) > max_update_text_bytes:
            raise StreamingAdapterUnavailable("streaming update exceeded bound")
        await _within_stream_deadline(
            websocket.send_json(
                {
                    "type": "update",
                    "sequence": update.sequence,
                    "text": update.text,
                    "stable_prefix": update.stable_prefix,
                    "final": update.final,
                }
            ),
            deadline,
        )


async def _close_stream_session(session: StreamingSpeechSession) -> None:
    try:
        async with asyncio.timeout(0.05):
            await asyncio.shield(session.close())
    except (TimeoutError, RuntimeError):
        return


def _json_object(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        raise ValueError("stream message must be text JSON")
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("stream message is invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("stream message must be JSON object")
    return payload


def _resolve_settings(settings: SpeechSettings | None) -> tuple[SpeechSettings, bool]:
    if settings is not None:
        return settings, False
    try:
        return SpeechSettings.model_validate({}), False
    except ValidationError:
        return (
            SpeechSettings.model_construct(
                runtime=SpeechRuntimeSettings(),
                faster_whisper=FasterWhisperSettings(),
                groq=None,
                models=SpeechModelSettings(),
                access=SpeechAccessSettings.model_construct(mode="internal", bearer_key=None),
            ),
            True,
        )


def _build_runtime(settings: SpeechSettings, settings_failed: bool) -> SpeechRuntimePort:
    if settings_failed:
        return UnavailableSpeechRuntime()
    try:
        return LazySpeechRuntime(settings)
    except SpeechConfigurationError:
        return UnavailableSpeechRuntime()


def _streaming_runtime(runtime: SpeechRuntimePort) -> StreamingRuntimePort | None:
    open_stream = getattr(runtime, "open_stream", None)
    if not callable(open_stream):
        return None
    return cast(StreamingRuntimePort, runtime)


async def _parse_source(request: Request, settings: SpeechSettings) -> SpeechFile:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > settings.runtime.max_upload_bytes + _MULTIPART_RESERVE_BYTES:
                raise DomainError(ErrorCode.PAYLOAD_TOO_LARGE)
        except ValueError:
            raise DomainError(ErrorCode.INVALID_REQUEST) from None
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().casefold()
    if media_type != "multipart/form-data":
        raise DomainError(ErrorCode.UNSUPPORTED_MEDIA)

    form: FormData | None = None
    try:
        try:
            form = await request.form(
                max_files=2,
                max_fields=2,
                max_part_size=settings.runtime.max_upload_bytes + 1,
            )
        except (MultiPartException, StarletteHTTPException, ValueError) as error:
            if "maximum size" in str(error).casefold():
                raise DomainError(ErrorCode.PAYLOAD_TOO_LARGE) from error
            raise DomainError(ErrorCode.INVALID_REQUEST) from error
        assert form is not None
        items = form.multi_items()
        values = dict(items)
        if len(items) != len(values) or set(values) not in ({"file"}, {"file", "metadata"}):
            raise DomainError(ErrorCode.INVALID_REQUEST)
        upload = values["file"]
        if not isinstance(upload, UploadFile):
            raise DomainError(ErrorCode.INVALID_REQUEST)
        metadata = _parse_metadata(values.get("metadata"))
        content = await upload.read(settings.runtime.max_upload_bytes + 1)
        if len(content) > settings.runtime.max_upload_bytes:
            raise DomainError(ErrorCode.PAYLOAD_TOO_LARGE)
        if not content or not upload.filename:
            raise DomainError(ErrorCode.INVALID_AUDIO)
        return SpeechFile(
            filename=upload.filename,
            content_type=upload.content_type or "",
            content=content,
            metadata=metadata,
        )
    finally:
        if form is not None:
            await form.close()


def _parse_metadata(value: object) -> SpeechMetadata:
    if value is None:
        return SpeechMetadata()
    if not isinstance(value, str):
        raise DomainError(ErrorCode.INVALID_REQUEST)
    try:
        parsed = json.loads(value)
        return SpeechMetadata.model_validate(parsed)
    except (TypeError, ValueError, ValidationError) as error:
        raise DomainError(ErrorCode.INVALID_REQUEST) from error


def _request_is_authorized(request: Request, settings: SpeechSettings) -> bool:
    if settings.access.mode == "internal" or request.url.path == "/health/live":
        return True
    return _matches_bearer_key(request.headers.getlist("authorization"), settings)


def _websocket_is_authorized(websocket: WebSocket, settings: SpeechSettings) -> bool:
    if settings.access.mode == "internal":
        return True
    return _matches_bearer_key(websocket.headers.getlist("authorization"), settings)


def _matches_bearer_key(authorizations: list[str], settings: SpeechSettings) -> bool:
    if len(authorizations) != 1:
        return False
    authorization = authorizations[0]
    if authorization.count(" ") != 1:
        return False
    scheme, presented_key = authorization.split(" ", 1)
    configured_key = settings.access.bearer_key
    if scheme != "Bearer" or configured_key is None or not presented_key:
        return False
    try:
        return hmac.compare_digest(
            presented_key.encode("ascii"), configured_key.get_secret_value().encode("ascii")
        )
    except UnicodeEncodeError:
        return False


def _runtime_attestation(settings: SpeechSettings, runtime: SpeechRuntimePort) -> dict[str, object]:
    asr, diarization = runtime.model_revisions()
    attestation_runtime: dict[str, object] = {
        "device": settings.runtime.device,
        "compute_type": settings.faster_whisper.compute_type(device=settings.runtime.device),
        "asr": _component_attestation(asr),
        "diarization": _component_attestation(diarization),
    }
    if settings.runtime.image_digest is not None:
        attestation_runtime["image_digest"] = settings.runtime.image_digest
    return {
        "runtime": attestation_runtime,
    }


def _component_attestation(component: ComponentRevision) -> dict[str, str]:
    return {
        "package": component.package,
        "package_version": component.package_version,
        "model_id": component.model_id,
        "model_revision": component.model_revision,
    }


def _request_id_from_header(value: str | None) -> UUID:
    if value is not None:
        try:
            return UUID(value)
        except ValueError:
            pass
    return uuid4()
