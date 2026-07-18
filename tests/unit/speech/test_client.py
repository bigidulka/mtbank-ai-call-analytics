from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from pydantic import HttpUrl, SecretStr, TypeAdapter, ValidationError

from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RoleResolution,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.speech.client import HttpSpeechServiceClient, SpeechServiceClientSettings
from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse

_SAFE_REMOTE_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class FailingReadStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.read_started = False
        self._chunks: tuple[bytes, ...] = ()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.read_started = True
        for chunk in self._chunks:
            yield chunk
        raise AssertionError("response body must not be read")

    async def aclose(self) -> None:
        return None


def _stream_response(
    status_code: int,
    request: httpx.Request,
    *,
    chunks: tuple[bytes, ...],
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status_code, headers=headers, stream=AsyncChunks(chunks), request=request)


def _json_response(status_code: int, payload: object, request: httpx.Request) -> httpx.Response:
    return _stream_response(
        status_code,
        request,
        chunks=(json.dumps(payload, separators=(",", ":")).encode("utf-8"),),
        headers={"content-type": "application/json"},
    )


def _url(value: str) -> HttpUrl:
    return TypeAdapter(HttpUrl).validate_python(value)


def _response() -> SpeechTranscriptionResponse:
    segment_id = UUID("22222222-2222-4222-8222-222222222222")
    revision = ComponentRevision(
        package="test-package",
        package_version="1.0.0",
        model_id="test-model",
        model_revision="test/v1",
    )
    return SpeechTranscriptionResponse(
        transcript=TranscriptSnapshot(
            transcript_id=UUID("33333333-3333-4333-8333-333333333333"),
            audio_sha256="a" * 64,
            revision="transcript/v1",
            language="ru",
            duration_seconds=1.0,
            segments=(
                TranscriptSegment(
                    id=segment_id,
                    original_speaker_id="speaker-1",
                    speaker=SpeakerRole.OPERATOR,
                    role_confidence=0.9,
                    start=0.0,
                    end=1.0,
                    text="Добрый день.",
                    redacted_text="Добрый день.",
                ),
            ),
            role_resolution=RoleResolution(
                assignments=(
                    RoleAssignment(
                        original_speaker_id="speaker-1",
                        role=SpeakerRole.OPERATOR,
                        confidence=0.9,
                        evidence_segment_ids=(segment_id,),
                    ),
                ),
                needs_review=False,
            ),
            asr_metadata=ASRMetadata(
                asr=revision,
                alignment=revision,
                diarization=revision,
                language="ru",
                processing_ms=1,
            ),
            created_at=datetime(2026, 7, 17, tzinfo=UTC),
        )
    )


def _source() -> SpeechFile:
    return SpeechFile(filename="call.wav", content_type="audio/wav", content=b"RIFF")


def test_internal_speech_request_has_no_authorization_and_disables_proxy_and_redirects() -> None:
    requests: list[httpx.Request] = []
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _json_response(200, _response().model_dump(mode="json"), request)

    def client_factory(*, timeout: float, trust_env: bool, follow_redirects: bool) -> httpx.AsyncClient:
        captured.update(timeout=timeout, trust_env=trust_env, follow_redirects=follow_redirects)
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )

    client = HttpSpeechServiceClient(
        SpeechServiceClientSettings(base_url=_url("http://speech:8010")),
        client_factory=client_factory,
    )

    response = asyncio.run(client.transcribe(_source()))

    assert response == _response()
    assert len(requests) == 1
    assert requests[0].url == httpx.URL("http://speech:8010/v1/transcribe")
    assert "authorization" not in requests[0].headers
    assert requests[0].headers["accept-encoding"] == "identity"
    assert captured == {"timeout": 180.0, "trust_env": False, "follow_redirects": False}


def test_remote_speech_request_sends_bearer_only_for_remote_mode_without_exposing_key() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _json_response(200, _response().model_dump(mode="json"), request)

    def client_factory(*, timeout: float, trust_env: bool, follow_redirects: bool) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )

    settings = SpeechServiceClientSettings(
        mode="remote_https",
        base_url=_url("https://speech.example.test/api"),
        api_key=SecretStr(_SAFE_REMOTE_KEY),
        transcription_path="/v1/transcribe",
    )
    client = HttpSpeechServiceClient(settings, client_factory=client_factory)

    asyncio.run(client.transcribe(_source()))

    assert len(requests) == 1
    assert requests[0].url == httpx.URL("https://speech.example.test/api/v1/transcribe")
    assert requests[0].headers["authorization"] == f"Bearer {_SAFE_REMOTE_KEY}"
    assert requests[0].headers["accept-encoding"] == "identity"
    assert _SAFE_REMOTE_KEY not in repr(settings)
    assert _SAFE_REMOTE_KEY not in settings.model_dump_json()


def test_speech_client_rejects_redirect_without_following_it() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://redirect.example.test/transcribe"}, request=request)

    def client_factory(*, timeout: float, trust_env: bool, follow_redirects: bool) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )

    client = HttpSpeechServiceClient(
        SpeechServiceClientSettings(
            mode="remote_https",
            base_url=_url("https://speech.example.test"),
            api_key=SecretStr(_SAFE_REMOTE_KEY),
        ),
        client_factory=client_factory,
    )

    with pytest.raises(DomainError) as error:
        asyncio.run(client.transcribe(_source()))

    assert error.value.code is ErrorCode.SERVICE_UNAVAILABLE
    assert len(requests) == 1


def _upstream_error_payload(code: ErrorCode) -> dict[str, object]:
    return {
        "error": {
            "code": code.value,
            "message": "untrusted provider detail",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "retryable": False,
        }
    }


def _remote_client(
    handler: httpx.MockTransport,
    *,
    max_success_response_bytes: int = 4 * 1024 * 1024,
    max_error_response_bytes: int = 16 * 1024,
) -> HttpSpeechServiceClient:
    def client_factory(*, timeout: float, trust_env: bool, follow_redirects: bool) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=handler,
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )

    return HttpSpeechServiceClient(
        SpeechServiceClientSettings(
            mode="remote_https",
            base_url=_url("https://speech.example.test"),
            api_key=SecretStr(_SAFE_REMOTE_KEY),
            max_success_response_bytes=max_success_response_bytes,
            max_error_response_bytes=max_error_response_bytes,
        ),
        client_factory=client_factory,
    )


@pytest.mark.parametrize(
    ("status_code", "upstream_code"),
    (
        (401, ErrorCode.UNAUTHENTICATED),
        (403, ErrorCode.FORBIDDEN),
        (500, ErrorCode.INTERNAL_ERROR),
    ),
)
def test_remote_client_sanitizes_untrusted_auth_and_server_errors(
    status_code: int,
    upstream_code: ErrorCode,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(status_code, _upstream_error_payload(upstream_code), request)

    with pytest.raises(DomainError) as error:
        asyncio.run(_remote_client(httpx.MockTransport(handler)).transcribe(_source()))

    assert error.value.code is ErrorCode.PROVIDER_FAILURE


def test_remote_client_rejects_status_and_code_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(401, _upstream_error_payload(ErrorCode.ROLE_RESOLUTION_REQUIRED), request)

    with pytest.raises(DomainError) as error:
        asyncio.run(_remote_client(httpx.MockTransport(handler)).transcribe(_source()))

    assert error.value.code is ErrorCode.PROVIDER_FAILURE


def test_remote_client_propagates_only_matching_speech_safe_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(422, _upstream_error_payload(ErrorCode.NO_SPEECH), request)

    with pytest.raises(DomainError) as error:
        asyncio.run(_remote_client(httpx.MockTransport(handler)).transcribe(_source()))

    assert error.value.code is ErrorCode.NO_SPEECH


def test_remote_client_sanitizes_malformed_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(422, {"detail": "untrusted"}, request)

    with pytest.raises(DomainError) as error:
        asyncio.run(_remote_client(httpx.MockTransport(handler)).transcribe(_source()))

    assert error.value.code is ErrorCode.SERVICE_UNAVAILABLE


def test_client_rejects_oversized_success_response_while_reading_raw_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(200, request, chunks=(b"a" * 8, b"b" * 9))

    with pytest.raises(DomainError) as error:
        asyncio.run(
            _remote_client(
                httpx.MockTransport(handler),
                max_success_response_bytes=16,
                max_error_response_bytes=8,
            ).transcribe(_source())
        )

    assert error.value.code is ErrorCode.SERVICE_UNAVAILABLE


def test_client_rejects_oversized_error_response_while_reading_raw_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(500, request, chunks=(b"a" * 8, b"b" * 9))

    with pytest.raises(DomainError) as error:
        asyncio.run(
            _remote_client(
                httpx.MockTransport(handler),
                max_success_response_bytes=32,
                max_error_response_bytes=16,
            ).transcribe(_source())
        )

    assert error.value.code is ErrorCode.PROVIDER_FAILURE


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    ((200, ErrorCode.SERVICE_UNAVAILABLE), (500, ErrorCode.PROVIDER_FAILURE)),
)
def test_client_rejects_oversized_content_length_without_reading_body(
    status_code: int,
    expected_code: ErrorCode,
) -> None:
    stream = FailingReadStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"content-length": "17"},
            stream=stream,
            request=request,
        )

    with pytest.raises(DomainError) as error:
        asyncio.run(
            _remote_client(
                httpx.MockTransport(handler),
                max_success_response_bytes=16,
                max_error_response_bytes=8,
            ).transcribe(_source())
        )

    assert error.value.code is expected_code
    assert not stream.read_started


def test_client_rejects_gzip_response_before_reading_or_decompression() -> None:
    stream = FailingReadStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-length": "1"},
            stream=stream,
            request=request,
        )

    with pytest.raises(DomainError) as error:
        asyncio.run(_remote_client(httpx.MockTransport(handler)).transcribe(_source()))

    assert error.value.code is ErrorCode.SERVICE_UNAVAILABLE
    assert not stream.read_started


def test_client_settings_require_bounded_distinct_success_and_error_limits() -> None:
    settings = SpeechServiceClientSettings(
        base_url=_url("http://speech:8010"),
        max_success_response_bytes=64,
        max_error_response_bytes=16,
    )

    assert settings.max_success_response_bytes == 64
    assert settings.max_error_response_bytes == 16
    with pytest.raises(ValidationError):
        SpeechServiceClientSettings(
            base_url=_url("http://speech:8010"),
            max_success_response_bytes=16,
            max_error_response_bytes=16,
        )
