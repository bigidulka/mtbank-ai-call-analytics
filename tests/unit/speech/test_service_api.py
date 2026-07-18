from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
from pydantic import SecretStr

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
from services.speech.app import create_app
from services.speech.errors import MediaTimeoutError, NoSpeechError, SpeechOverloadedError
from services.speech.runtime import LazySpeechRuntime
from services.speech.settings import GroqTranscriptionSettings, SpeechRuntimeSettings, SpeechSettings
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


def test_transcribe_api_parses_explicit_role_metadata_and_has_stable_contract() -> None:
    async def scenario() -> None:
        runtime = StubRuntime()
        response = await _request(
            create_app(runtime=runtime),
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
                create_app(runtime=StubRuntime(error=error)),
                "POST",
                "/v1/transcribe",
                files={"file": ("call.wav", b"RIFFxxxxWAVE", "audio/wav")},
            )
            assert response.status_code == status
            assert response.json()["error"]["code"] == code

        malformed = await _request(
            create_app(runtime=StubRuntime()),
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
