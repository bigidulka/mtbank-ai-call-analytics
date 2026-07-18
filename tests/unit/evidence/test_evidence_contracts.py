from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from mtbank_ai.domain.events import (
    EventAttribute,
    LifecycleEventType,
    RedactedPayload,
    RunEvent,
    RunStatus,
)
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.evidence.bundle import ArtifactDigest, EvidenceBundleManifest, EvidenceReference
from mtbank_ai.evidence.envelope import (
    MediaDescriptor,
    ModelBinding,
    PrivacyPolicy,
    ProviderFingerprint,
    RevisionSet,
    RunBudget,
    RunEnvelope,
    RunSource,
)

RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
REQUEST_ID = UUID("22222222-2222-4222-8222-222222222222")
CORRELATION_ID = UUID("33333333-3333-4333-8333-333333333333")
NOW = datetime(2026, 7, 15, tzinfo=UTC)


def _speech_revision(model_id: str, revision: str) -> ComponentRevision:
    return ComponentRevision(
        package="speech-package",
        package_version="1.0.0",
        model_id=model_id,
        model_revision=revision,
    )


def _envelope(**changes: object) -> RunEnvelope:
    values: dict[str, object] = {
        "run_id": RUN_ID,
        "request_id": REQUEST_ID,
        "correlation_id": CORRELATION_ID,
        "source": RunSource.REST_FILE,
        "input_media": MediaDescriptor(
            sha256="a" * 64,
            mime_type="audio/wav",
            duration_seconds=12.5,
            sample_rate_hz=16_000,
            channels=1,
        ),
        "provider": ProviderFingerprint(
            model_bindings=(
                ModelBinding(
                    agent_id="classifier",
                    provider_id="openai-compatible",
                    model_id="configured-model",
                    reasoning_effort="high",
                ),
            ),
        ),
        "revisions": RevisionSet(
            code_sha="abcdef0",
            prompt_bundle_hash="c" * 64,
            taxonomy_version="taxonomy/v1",
            quality_rubric_version="quality/v1",
            compliance_policy_version="compliance/v1",
            asr=_speech_revision("large-v3", "asr/v1"),
            alignment=_speech_revision("wav2vec2", "alignment/v1"),
            diarization=_speech_revision("speaker-diarization", "diarization/v1"),
        ),
        "budget": RunBudget(
            deadline_at=NOW + timedelta(minutes=2),
            max_llm_turns=3,
            max_total_tokens=10_000,
            max_cost_usd=Decimal("1.50"),
        ),
        "privacy": PrivacyPolicy(
            mode="redacted-cloud",
            raw_audio_retention_seconds=0,
            evidence_retention_days=30,
            allow_full_content_evidence=False,
        ),
        "created_at": NOW,
    }
    values.update(changes)
    return RunEnvelope.model_validate(values)


def test_run_envelope_is_strict_frozen_and_contains_no_endpoint_or_secret_field() -> None:
    envelope = _envelope()

    with pytest.raises(ValidationError, match="frozen"):
        envelope.source = RunSource.EVAL
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunEnvelope.model_validate({**envelope.model_dump(), "api_key": "not-allowed"})
    payload = envelope.model_dump(mode="json")
    assert payload["schema_version"] == "1"
    assert payload["provider"]["model_bindings"][0]["provider_id"] == "openai-compatible"
    assert payload["provider"]["model_bindings"][0]["reasoning_effort"] == "high"
    assert "base_url" not in payload
    assert "api_key" not in payload


def test_run_envelope_rejects_invalid_numbers_deadline_and_utc() -> None:
    with pytest.raises(ValidationError):
        _envelope(
            input_media=MediaDescriptor(
                sha256="a" * 64,
                mime_type="audio/wav",
                duration_seconds=float("nan"),
                sample_rate_hz=16_000,
                channels=1,
            )
        )
    with pytest.raises(ValidationError, match="deadline"):
        _envelope(
            budget=RunBudget(
                deadline_at=NOW,
                max_llm_turns=3,
                max_total_tokens=100,
                max_cost_usd=Decimal("0"),
            )
        )
    with pytest.raises(ValidationError, match="UTC"):
        _envelope(created_at=datetime(2026, 7, 15))


def test_eval_case_requires_dataset_and_model_bindings_are_unique() -> None:
    with pytest.raises(ValidationError, match="dataset_version"):
        RevisionSet(
            code_sha="abcdef0",
            prompt_bundle_hash="c" * 64,
            taxonomy_version="taxonomy/v1",
            quality_rubric_version="quality/v1",
            compliance_policy_version="compliance/v1",
            asr=_speech_revision("large-v3", "asr/v1"),
            alignment=_speech_revision("wav2vec2", "alignment/v1"),
            diarization=_speech_revision("speaker-diarization", "diarization/v1"),
            eval_case_id="case-1",
        )
    binding = ModelBinding(agent_id="classifier", provider_id="provider", model_id="model")
    with pytest.raises(ValidationError, match="agent_id"):
        ProviderFingerprint(model_bindings=(binding, binding))


def test_event_hash_chain_status_and_payload_are_immutable() -> None:
    first = RunEvent(
        run_id=RUN_ID,
        sequence=1,
        event_type=LifecycleEventType.RUN_STARTED,
        occurred_at=NOW,
        component="workflow",
        payload=RedactedPayload(fields=(EventAttribute(key="attempt", value=1),)),
        previous_hash=None,
        current_hash="d" * 64,
    )

    assert first.payload.fields[0].value == 1
    assert first.model_dump(mode="json")["payload"] == {"fields": [{"key": "attempt", "value": 1}]}
    assert tuple(RunStatus) == (
        RunStatus.QUEUED,
        RunStatus.PROCESSING,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
    )
    with pytest.raises(ValidationError, match="ключи"):
        RedactedPayload(
            fields=(
                EventAttribute(key="attempt", value=1),
                EventAttribute(key="attempt", value=2),
            )
        )
    with pytest.raises(ValidationError, match="previous_hash"):
        RunEvent(
            run_id=RUN_ID,
            sequence=2,
            event_type=LifecycleEventType.RUN_COMPLETED,
            occurred_at=NOW,
            component="workflow",
            payload=RedactedPayload(fields=()),
            previous_hash=None,
            current_hash="e" * 64,
        )


def test_evidence_and_artifact_collections_require_unique_identifiers() -> None:
    segment_id = UUID("44444444-4444-4444-8444-444444444444")
    with pytest.raises(ValidationError, match="уникальны"):
        EvidenceReference(segment_ids=(segment_id, segment_id))

    artifact = ArtifactDigest(
        name="events.jsonl",
        media_type="application/json",
        size_bytes=10,
        sha256="f" * 64,
    )
    with pytest.raises(ValidationError, match="имена"):
        EvidenceBundleManifest(
            run_id=RUN_ID,
            envelope_sha256="a" * 64,
            events_sha256="b" * 64,
            artifacts=(artifact, artifact),
            created_at=NOW,
        )
