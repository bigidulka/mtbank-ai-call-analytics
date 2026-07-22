from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from starlette.websockets import WebSocketDisconnect

from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RoleResolution,
    RoleResolutionSource,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.speech.contracts import RoleResolutionCandidate, SpeechFile, SpeechTranscriptionResponse
from mtbank_ai.speech.roles import RoleResolutionRequiredError
from services.speech.app import _matches_bearer_key, create_app
from services.speech.errors import MediaTimeoutError, NoSpeechError, SpeechOverloadedError
from services.speech.runtime import LazySpeechRuntime
from services.speech.settings import (
    GroqTranscriptionSettings,
    SpeechAccessSettings,
    SpeechRuntimeSettings,
    SpeechSettings,
)
from tests.unit.speech._helpers import make_registry


class StubRuntime:
    def __init__(self, *, response: SpeechTranscriptionResponse | None = None, error: Exception | None = None) -> None:
        self.response = response or _response()
        self.error = error
        self.sources: list[SpeechFile] = []
        self.is_ready = True

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        self.sources.append(source)
        if self.error is not None:
            raise self.error
        return self.response

    async def ready(self) -> bool:
        return self.is_ready

    def model_revisions(self) -> tuple[ComponentRevision, ComponentRevision]:
        return _revision("faster-whisper", "verified-asr"), _revision("pyannote.audio", "verified-diarization")

    async def close(self) -> None:
        return None


def _response() -> SpeechTranscriptionResponse:
    segment_id = UUID("11111111-1111-4111-8111-111111111111")
    snapshot = TranscriptSnapshot(
        transcript_id=UUID("22222222-2222-4222-8222-222222222222"),
        audio_sha256="a" * 64,
        revision="speech/test",
        language="ru",
        duration_seconds=1.0,
        segments=(
            TranscriptSegment(
                id=segment_id,
                original_speaker_id="SPEAKER_00",
                speaker=SpeakerRole.OPERATOR,
                role_confidence=1.0,
                start=0.0,
                end=1.0,
                text="Добрый день.",
                redacted_text="Добрый день.",
            ),
        ),
        role_resolution=RoleResolution(
            assignments=(
                RoleAssignment(
                    original_speaker_id="SPEAKER_00",
                    role=SpeakerRole.OPERATOR,
                    confidence=1.0,
                    evidence_segment_ids=(segment_id,),
                    source=RoleResolutionSource.METADATA,
                    resolution_evidence="test",
                ),
            ),
            needs_review=False,
        ),
        asr_metadata=ASRMetadata(
            asr=_revision("faster-whisper", "medium"),
            alignment=_revision("whisperx", "alignment"),
            diarization=_revision("pyannote.audio", "community-1"),
            language="ru",
            processing_ms=1,
        ),
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )
    return SpeechTranscriptionResponse(transcript=snapshot)


def _revision(package: str, model_id: str) -> ComponentRevision:
    return ComponentRevision(
        package=package,
        package_version="test",
        model_id=model_id,
        model_revision="test",
        artifact_sha256="b" * 64,
    )


async def _request(app, method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://speech.test") as client:
        return await client.request(method, path, **kwargs)


def _internal_settings() -> SpeechSettings:
    return SpeechSettings(access=SpeechAccessSettings(mode="internal"))


def test_transcribe_api_parses_explicit_role_metadata_and_has_stable_contract() -> None:
    async def scenario() -> None:
        runtime = StubRuntime()
        response = await _request(
            create_app(settings=_internal_settings(), runtime=runtime),
            "POST",
            "/v1/transcribe",
            files={"file": ("call.wav", b"RIFFxxxxWAVE", "audio/wav")},
            data={
                "metadata": json.dumps(
                    {
                        "role_mappings": [
                            {
                                "original_speaker_id": "SPEAKER_00",
                                "role": "Оператор",
                                "confidence": 1.0,
                                "evidence": "metadata/test",
                            }
                        ]
                    }
                )
            },
            headers={"X-Request-ID": "33333333-3333-4333-8333-333333333333"},
        )

        assert response.status_code == 200
        assert response.headers["x-request-id"] == "33333333-3333-4333-8333-333333333333"
        assert response.json()["transcript"]["segments"][0]["speaker"] == "Оператор"
        assert runtime.sources[0].metadata.role_mappings[0].original_speaker_id == "SPEAKER_00"

    asyncio.run(scenario())


def test_speech_api_maps_no_speech_timeout_queue_and_invalid_metadata() -> None:
    async def scenario() -> None:
        cases = (
            (NoSpeechError("no speech"), 422, "no_speech"),
            (
                RoleResolutionRequiredError(
                    (
                        RoleResolutionCandidate(
                            original_speaker_id="SPEAKER_00",
                            evidence_segment_ids=(UUID("11111111-1111-4111-8111-111111111111"),),
                        ),
                    )
                ),
                409,
                "role_resolution_required",
            ),
            (MediaTimeoutError("timeout"), 504, "deadline_exceeded"),
            (SpeechOverloadedError("full"), 429, "quota_exceeded"),
        )
        for error, status, code in cases:
            response = await _request(
                create_app(settings=_internal_settings(), runtime=StubRuntime(error=error)),
                "POST",
                "/v1/transcribe",
                files={"file": ("call.wav", b"RIFFxxxxWAVE", "audio/wav")},
            )
            assert response.status_code == status
            assert response.json()["error"]["code"] == code

        malformed = await _request(
            create_app(settings=_internal_settings(), runtime=StubRuntime()),
            "POST",
            "/v1/transcribe",
            files={"file": ("call.wav", b"RIFFxxxxWAVE", "audio/wav")},
            data={"metadata": "[]"},
        )
        assert malformed.status_code == 422
        assert malformed.json()["error"]["code"] == "invalid_request"

    asyncio.run(scenario())


def test_speech_api_enforces_upload_limit_and_fail_closed_readiness(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = StubRuntime()
        runtime.is_ready = False
        settings = SpeechSettings(
            runtime=SpeechRuntimeSettings(max_upload_bytes=4, temp_root=str(tmp_path / "work")),
            groq=GroqTranscriptionSettings(api_key=SecretStr("test-groq-key")),
            access=SpeechAccessSettings(mode="internal"),
        )
        app = create_app(settings=settings, runtime=runtime)
        too_large = await _request(
            app,
            "POST",
            "/v1/transcribe",
            files={"file": ("call.wav", b"12345", "audio/wav")},
        )
        readiness = await _request(app, "GET", "/health/ready")
        live = await _request(app, "GET", "/health/live")

        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "payload_too_large"
        assert runtime.sources == []
        assert readiness.status_code == 503
        assert live.json() == {"status": "ok"}

    asyncio.run(scenario())


def test_speech_api_rejects_declared_and_chunked_oversized_multipart_before_spooling(tmp_path: Path) -> None:
    async def scenario() -> None:
        streamed_chunks: list[str] = []

        async def oversized_multipart() -> AsyncIterator[bytes]:
            streamed_chunks.append("prefix")
            yield b'--limit\r\nContent-Disposition: form-data; name="file"; filename="call.wav"\r\n\r\n'
            streamed_chunks.append("oversized")
            yield b"x" * (64 * 1024 + 8)
            streamed_chunks.append("tail")
            yield b"\r\n--limit--\r\n"

        runtime = StubRuntime()
        settings = SpeechSettings(
            runtime=SpeechRuntimeSettings(max_upload_bytes=4, temp_root=str(tmp_path / "work")),
            groq=GroqTranscriptionSettings(api_key=SecretStr("test-groq-key")),
            access=SpeechAccessSettings(mode="internal"),
        )
        app = create_app(settings=settings, runtime=runtime)
        headers = {"Content-Type": "multipart/form-data; boundary=limit"}
        declared = await _request(
            app,
            "POST",
            "/v1/transcribe",
            content=b"x" * (64 * 1024 + 8),
            headers=headers,
        )
        declared_alias = await _request(
            app,
            "POST",
            "/v1/transcribe/",
            content=b"x" * (64 * 1024 + 8),
            headers=headers,
        )
        chunked = await _request(
            app,
            "POST",
            "/v1/transcribe",
            content=oversized_multipart(),
            headers={**headers, "Transfer-Encoding": "chunked"},
        )
        redirect = await _request(
            app,
            "POST",
            "/v1/transcribe/",
            content=b"",
            headers=headers,
        )

        for response in (declared, declared_alias, chunked):
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "payload_too_large"
        assert streamed_chunks == ["prefix", "oversized"]
        assert runtime.sources == []
        assert redirect.status_code == 307
        assert redirect.headers["location"].endswith("/v1/transcribe")

    asyncio.run(scenario())


def test_bearer_access_protects_remote_speech_boundary_and_keeps_live_anonymous(tmp_path: Path) -> None:
    async def scenario() -> None:
        key = "X2v9Qa7Lm4Rc8Nd6Hs3Kp5Zw1By0TeUf"
        settings = SpeechSettings(
            runtime=SpeechRuntimeSettings(
                temp_root=str(tmp_path / "work"), image_digest="sha256:" + "a" * 64
            ),
            access=SpeechAccessSettings(mode="bearer", bearer_key=SecretStr(key)),
        )
        app = create_app(settings=settings, runtime=StubRuntime())

        live = await _request(app, "GET", "/health/live")
        ready_missing = await _request(app, "GET", "/health/ready")
        ready_wrong = await _request(
            app, "GET", "/health/ready", headers={"Authorization": "Bearer " + "Z9m4Kb1Vx7Qa3Ln8Rc6Hs0Dw2Py5TeUf"}
        )
        ready_duplicate = await _request(
            app,
            "GET",
            "/health/ready",
            headers=[("Authorization", f"Bearer {key}"), ("Authorization", f"Bearer {key}")],
        )
        ready = await _request(app, "GET", "/health/ready", headers={"Authorization": f"Bearer {key}"})
        runtime = await _request(app, "GET", "/v1/runtime", headers={"Authorization": f"Bearer {key}"})

        assert live.status_code == 200
        for response in (ready_missing, ready_wrong, ready_duplicate):
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "unauthenticated"
            assert response.headers["www-authenticate"] == "Bearer"
            assert key not in response.text
        assert ready.status_code == 200
        assert runtime.status_code == 200
        expected_asr = _revision("faster-whisper", "verified-asr")
        expected_diarization = _revision("pyannote.audio", "verified-diarization")
        assert "artifact_sha256" not in runtime.text
        assert runtime.json() == {
            "runtime": {
                "device": "cpu",
                "compute_type": "int8",
                "image_digest": "sha256:" + "a" * 64,
                "asr": {
                    "package": expected_asr.package,
                    "package_version": expected_asr.package_version,
                    "model_id": expected_asr.model_id,
                    "model_revision": expected_asr.model_revision,
                },
                "diarization": {
                    "package": expected_diarization.package,
                    "package_version": expected_diarization.package_version,
                    "model_id": expected_diarization.model_id,
                    "model_revision": expected_diarization.model_revision,
                },
            }
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("image_digest", ("latest", "sha256:" + "A" * 64, "sha256:" + "a" * 63))
def test_runtime_image_digest_requires_immutable_lowercase_sha256(image_digest: str) -> None:
    with pytest.raises(ValidationError, match="image_digest"):
        SpeechRuntimeSettings(image_digest=image_digest)


def test_bearer_access_rejects_non_ascii_configured_key_before_auth_matching() -> None:
    with pytest.raises(ValidationError, match="ASCII") as error:
        SpeechAccessSettings(mode="bearer", bearer_key=SecretStr("X2v9Qa7Lm4Rc8Nd6Hs3Kp5Zw1By0Te-к"))

    assert "Te-к" not in str(error.value)


def test_bearer_access_fails_closed_before_transcribe_body_parsing(tmp_path: Path) -> None:
    async def scenario() -> None:
        key = "X2v9Qa7Lm4Rc8Nd6Hs3Kp5Zw1By0TeUf"
        settings = SpeechSettings(
            runtime=SpeechRuntimeSettings(temp_root=str(tmp_path / "work")),
            access=SpeechAccessSettings(mode="bearer", bearer_key=SecretStr(key)),
        )
        runtime = StubRuntime()
        app = create_app(settings=settings, runtime=runtime)
        malformed_headers = (
            {"Authorization": "Basic " + key},
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer  " + key},
            [("Authorization", f"Bearer {key}"), ("Authorization", f"Bearer {key}")],
        )
        assert not _matches_bearer_key(["Bearer non-ascii-ключ"], settings)
        for headers in malformed_headers:
            response = await _request(
                app,
                "POST",
                "/v1/transcribe",
                content=b"not-a-multipart-body",
                headers=headers,
            )
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert runtime.sources == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "bearer_key"),
    ((None, None), ("bearer", None), ("bearer", "unsafe")),
)
def test_missing_or_invalid_access_settings_fail_closed_before_anonymous_body_parsing(
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
    bearer_key: str | None,
) -> None:
    if mode is None:
        monkeypatch.delenv("MTBANK_SPEECH__ACCESS__MODE", raising=False)
    else:
        monkeypatch.setenv("MTBANK_SPEECH__ACCESS__MODE", mode)
    if bearer_key is None:
        monkeypatch.delenv("MTBANK_SPEECH__ACCESS__BEARER_KEY", raising=False)
    else:
        monkeypatch.setenv("MTBANK_SPEECH__ACCESS__BEARER_KEY", bearer_key)
    app = create_app()

    async def scenario() -> None:
        streamed_chunks: list[str] = []

        async def multipart() -> AsyncIterator[bytes]:
            streamed_chunks.append("body")
            yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="call.wav"\r\n\r\n'

        live = await _request(app, "GET", "/health/live")
        ready = await _request(app, "GET", "/health/ready")
        runtime = await _request(app, "GET", "/v1/runtime")
        transcribe = await _request(
            app,
            "POST",
            "/v1/transcribe",
            content=multipart(),
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )

        assert live.status_code == 200
        for response in (ready, runtime, transcribe):
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "service_unavailable"
        assert streamed_chunks == []

    asyncio.run(scenario())

    with pytest.raises(WebSocketDisconnect) as error:
        with TestClient(app).websocket_connect("/v1/stream"):
            pass
    assert error.value.code == 1013


def test_cuda_lifespan_warms_in_background_without_blocking_liveness(tmp_path: Path) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    started = Event()
    release = Event()

    class WarmEngine:
        def warm(self) -> None:
            started.set()
            assert release.wait(timeout=3.0)

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, WarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        app = create_app(settings=settings, runtime=runtime)
        async with app.router.lifespan_context(app):
            readiness = await _request(app, "GET", "/health/ready")
            assert await asyncio.to_thread(started.wait, 1.0)
            streamed_chunks: list[str] = []

            async def multipart() -> AsyncIterator[bytes]:
                streamed_chunks.append("body")
                yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="call.wav"\r\n\r\n'

            transcribe = await _request(
                app,
                "POST",
                "/v1/transcribe",
                content=multipart(),
                headers={"Content-Type": "multipart/form-data; boundary=boundary"},
            )
            live = await _request(app, "GET", "/health/live")

            assert live.status_code == 200
            assert live.json() == {"status": "ok"}
            assert readiness.status_code == 503
            assert transcribe.status_code == 503
            assert streamed_chunks == []

            release.set()
            for _ in range(100):
                readiness = await _request(app, "GET", "/health/ready")
                if readiness.status_code == 200:
                    break
                await asyncio.sleep(0.01)
            assert readiness.status_code == 200

    asyncio.run(scenario())


def test_cuda_lifespan_logs_sanitized_background_warmup_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)

    class FailingWarmEngine:
        def warm(self) -> None:
            raise RuntimeError("secret model path /workspace/models")

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, FailingWarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        app = create_app(settings=settings, runtime=runtime)
        caplog.set_level(logging.ERROR, logger="services.speech.app")
        async with app.router.lifespan_context(app):
            for _ in range(100):
                if caplog.records:
                    break
                await asyncio.sleep(0.01)

    asyncio.run(scenario())

    assert [record.getMessage() for record in caplog.records] == ['{"event":"speech_cuda_warmup_failed"}']
    assert "secret" not in caplog.text
    assert "/workspace/models" not in caplog.text


def test_cuda_pending_warmup_closes_websocket_before_accept(tmp_path: Path) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    started = Event()
    release = Event()

    class WarmEngine:
        def warm(self) -> None:
            started.set()
            assert release.wait(timeout=3.0)

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, WarmEngine())

    runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
    with TestClient(create_app(settings=settings, runtime=runtime)) as client:
        assert started.wait(timeout=1.0)
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/v1/stream"):
                pass
        assert error.value.code == 1013
        release.set()

        for _ in range(100):
            if client.get("/health/ready").status_code == 200:
                break
            Event().wait(0.01)
        else:
            pytest.fail("CUDA warmup did not make readiness available")


def test_cuda_lifespan_cancels_inflight_background_warmup(tmp_path: Path) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    started = Event()
    release = Event()

    class WarmEngine:
        def warm(self) -> None:
            started.set()
            assert release.wait(timeout=3.0)

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, WarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        app = create_app(settings=settings, runtime=runtime)
        async with app.router.lifespan_context(app):
            assert await asyncio.to_thread(started.wait, 1.0)
        release.set()
        assert not await runtime.ready()

    asyncio.run(scenario())


def test_speech_readiness_rechecks_artifacts_after_a_successful_probe(tmp_path: Path) -> None:
    async def scenario() -> None:
        _, settings = make_registry(tmp_path)
        runtime = LazySpeechRuntime(settings)
        app = create_app(settings=settings, runtime=runtime)

        first = await _request(app, "GET", "/health/ready")
        (tmp_path / "artifacts" / "diarization" / "artifact.bin").write_bytes(b"tampered")
        second = await _request(app, "GET", "/health/ready")

        assert first.status_code == 200
        assert first.json() == {"status": "ready"}
        assert second.status_code == 503
        assert second.json()["error"]["code"] == "service_unavailable"

    asyncio.run(scenario())
