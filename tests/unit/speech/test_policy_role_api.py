from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import httpx

from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RolePolicyProvenance,
    RoleResolution,
    RoleResolutionSource,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse
from services.speech.app import create_app


class PolicyRuntime:
    def __init__(self) -> None:
        self._response = _response()

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        del source
        return self._response

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _response() -> SpeechTranscriptionResponse:
    policy = PolicyRegistry().roles
    provenance = RolePolicyProvenance(
        policy_id=policy.policy.metadata.policy_id,
        version=policy.version,
        owner=policy.owner,
        effective_date=policy.effective_date,
        sha256=policy.sha256,
    )
    operator_id = UUID("11111111-1111-4111-8111-111111111111")
    client_id = UUID("22222222-2222-4222-8222-222222222222")
    revision = ComponentRevision(
        package="test-package",
        package_version="1.0.0",
        model_id="test-model",
        model_revision="test/v1",
        artifact_sha256="a" * 64,
    )
    return SpeechTranscriptionResponse(
        transcript=TranscriptSnapshot(
            transcript_id=UUID("33333333-3333-4333-8333-333333333333"),
            audio_sha256="b" * 64,
            revision="speech/test",
            language="ru",
            duration_seconds=2.0,
            segments=(
                TranscriptSegment(
                    id=operator_id,
                    original_speaker_id="speaker-operator",
                    speaker=SpeakerRole.OPERATOR,
                    role_confidence=1.0,
                    start=0.0,
                    end=1.0,
                    text="МТБанк, оператор Анна.",
                    redacted_text="МТБанк, оператор Анна.",
                ),
                TranscriptSegment(
                    id=client_id,
                    original_speaker_id="speaker-client",
                    speaker=SpeakerRole.CLIENT,
                    role_confidence=0.9,
                    start=1.0,
                    end=2.0,
                    text="Хочу уточнить перевод.",
                    redacted_text="Хочу уточнить перевод.",
                ),
            ),
            role_resolution=RoleResolution(
                assignments=(
                    RoleAssignment(
                        original_speaker_id="speaker-operator",
                        role=SpeakerRole.OPERATOR,
                        confidence=1.0,
                        evidence_segment_ids=(operator_id,),
                        source=RoleResolutionSource.POLICY,
                        resolution_evidence=f"v1/{policy.sha256}",
                    ),
                    RoleAssignment(
                        original_speaker_id="speaker-client",
                        role=SpeakerRole.CLIENT,
                        confidence=0.9,
                        evidence_segment_ids=(client_id,),
                        source=RoleResolutionSource.POLICY,
                        resolution_evidence=f"v1/{policy.sha256}",
                    ),
                ),
                needs_review=False,
                policy_provenance=provenance,
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


def test_speech_api_serializes_policy_role_provenance_without_raw_evidence() -> None:
    async def scenario() -> None:
        policy = PolicyRegistry().roles
        transport = httpx.ASGITransport(app=create_app(runtime=PolicyRuntime()))
        async with httpx.AsyncClient(transport=transport, base_url="http://speech.test") as client:
            response = await client.post(
                "/v1/transcribe",
                files={"file": ("call.wav", b"RIFF", "audio/wav")},
            )

        assert response.status_code == 200
        payload = response.json()["transcript"]["role_resolution"]
        assert {item["source"] for item in payload["assignments"]} == {"policy"}
        assert payload["policy_provenance"] == {
            "policy_id": "roles",
            "version": "v1",
            "owner": policy.owner,
            "effective_date": policy.effective_date,
            "sha256": policy.sha256,
        }
        assert "МТБанк" not in payload["assignments"][0]["resolution_evidence"]

    asyncio.run(scenario())
