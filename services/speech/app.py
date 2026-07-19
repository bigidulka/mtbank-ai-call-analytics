"""Internal FastAPI boundary for the canonical batch speech service."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from mtbank_ai.api.body_limits import BodyLimitMiddleware
from mtbank_ai.api.error_handlers import install_error_handlers
from mtbank_ai.api.schemas import HealthResponse
from mtbank_ai.domain.errors import DomainError, ErrorCode, ErrorResponse
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
    SpeechModelSettings,
    SpeechRuntimeSettings,
    SpeechSettings,
)

_MULTIPART_RESERVE_BYTES = 64 * 1024
_TRANSCRIBE_PATHS = frozenset({"/v1/transcribe", "/v1/transcribe/"})

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
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
}


def create_app(
    settings: SpeechSettings | None = None,
    runtime: SpeechRuntimePort | None = None,
) -> FastAPI:
    resolved_settings, settings_failed = _resolve_settings(settings)
    resolved_runtime = runtime or _build_runtime(resolved_settings, settings_failed)
    resolved_streaming_runtime = _streaming_runtime(resolved_runtime)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        yield
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
        responses={503: {"model": ErrorResponse, "description": "Speech artifacts are unavailable"}},
        tags=["health"],
    )
    async def ready() -> HealthResponse:
        if not await resolved_runtime.ready():
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
        return HealthResponse(status="ready")

    @app.post(
        "/v1/transcribe",
        response_model=SpeechTranscriptionResponse,
        responses=_ERROR_RESPONSES,
        tags=["speech"],
    )
    async def transcribe(request: Request) -> SpeechTranscriptionResponse:
        source = await _parse_source(request, resolved_settings)
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


def _request_id_from_header(value: str | None) -> UUID:
    if value is not None:
        try:
            return UUID(value)
        except ValueError:
            pass
    return uuid4()
