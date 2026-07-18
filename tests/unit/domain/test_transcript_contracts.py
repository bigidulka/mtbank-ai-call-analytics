from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RoleResolution,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
    WordTimestamp,
)

SEGMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
TRANSCRIPT_ID = UUID("22222222-2222-4222-8222-222222222222")


def _segment(**changes: object) -> TranscriptSegment:
    values: dict[str, object] = {
        "id": SEGMENT_ID,
        "original_speaker_id": "SPEAKER_00",
        "speaker": SpeakerRole.OPERATOR,
        "role_confidence": 0.9,
        "start": 0.0,
        "end": 2.0,
        "text": "Добрый день.",
        "redacted_text": "Добрый день.",
        "word_timestamps": (
            WordTimestamp(word="Добрый", start=0.0, end=0.8, confidence=0.95),
            WordTimestamp(word="день", start=0.9, end=1.5, confidence=0.94),
        ),
    }
    values.update(changes)
    return TranscriptSegment.model_validate(values)


def _snapshot(**changes: object) -> TranscriptSnapshot:
    segment = _segment()
    values: dict[str, object] = {
        "transcript_id": TRANSCRIPT_ID,
        "audio_sha256": "a" * 64,
        "revision": "speech/v1",
        "language": "ru",
        "duration_seconds": 2.5,
        "segments": (segment,),
        "role_resolution": RoleResolution(
            assignments=(
                RoleAssignment(
                    original_speaker_id="SPEAKER_00",
                    role=SpeakerRole.OPERATOR,
                    confidence=0.9,
                    evidence_segment_ids=(SEGMENT_ID,),
                ),
            ),
            needs_review=False,
        ),
        "asr_metadata": ASRMetadata(
            asr=ComponentRevision(
                package="faster-whisper",
                package_version="1.0.0",
                model_id="large-v3",
                model_revision="model-revision",
                artifact_sha256="b" * 64,
            ),
            alignment=ComponentRevision(
                package="whisperx",
                package_version="3.0.0",
                model_id="wav2vec2",
                model_revision="alignment-revision",
            ),
            diarization=ComponentRevision(
                package="pyannote.audio",
                package_version="3.3.0",
                model_id="speaker-diarization",
                model_revision="diarization-revision",
            ),
            language="ru",
            processing_ms=125,
        ),
        "created_at": datetime(2026, 7, 15, tzinfo=UTC),
    }
    values.update(changes)
    return TranscriptSnapshot.model_validate(values)


def test_snapshot_is_frozen_strict_and_forbids_extra_fields() -> None:
    snapshot = _snapshot()

    with pytest.raises(ValidationError, match="frozen"):
        snapshot.language = "en"
    with pytest.raises(ValidationError, match="Extra inputs"):
        TranscriptSnapshot.model_validate({**snapshot.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        _snapshot(duration_seconds="2.5")
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.asr_metadata.asr.model_id = "other-model"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ComponentRevision.model_validate({**snapshot.asr_metadata.asr.model_dump(), "endpoint": "forbidden"})


def test_snapshot_rejects_non_finite_and_out_of_bounds_timestamps() -> None:
    with pytest.raises(ValidationError):
        _segment(end=float("inf"))
    with pytest.raises(ValidationError, match="duration"):
        _snapshot(segments=(_segment(end=3.0),))
    with pytest.raises(ValidationError, match="границы сегмента"):
        _segment(word_timestamps=(WordTimestamp(word="слово", start=1.9, end=2.1),))


def test_snapshot_requires_nondecreasing_unique_segments_but_allows_overlap() -> None:
    second_id = UUID("33333333-3333-4333-8333-333333333333")
    overlapping = _segment(id=second_id, start=1.0, end=2.2, word_timestamps=())
    first = _segment(start=0.0, end=1.5, word_timestamps=())
    resolution = RoleResolution(
        assignments=(
            RoleAssignment(
                original_speaker_id="SPEAKER_00",
                role=SpeakerRole.OPERATOR,
                confidence=0.9,
                evidence_segment_ids=(SEGMENT_ID, second_id),
            ),
        ),
        needs_review=False,
    )

    assert _snapshot(
        segments=(first, overlapping),
        duration_seconds=3.0,
        role_resolution=resolution,
    ).segments[1].start < first.end
    with pytest.raises(ValidationError, match="отсортированы"):
        _snapshot(segments=(overlapping, first), duration_seconds=3.0, role_resolution=resolution)
    with pytest.raises(ValidationError, match="уникальны"):
        _snapshot(segments=(_segment(), _segment()), duration_seconds=3.0)


def test_role_resolution_must_cover_and_match_segments() -> None:
    wrong_role = RoleResolution(
        assignments=(
            RoleAssignment(
                original_speaker_id="SPEAKER_00",
                role=SpeakerRole.CLIENT,
                confidence=0.9,
                evidence_segment_ids=(SEGMENT_ID,),
            ),
        ),
        needs_review=True,
    )
    with pytest.raises(ValidationError, match="не совпадает"):
        _snapshot(role_resolution=wrong_role)

    unknown_evidence = RoleResolution(
        assignments=(
            RoleAssignment(
                original_speaker_id="SPEAKER_00",
                role=SpeakerRole.OPERATOR,
                confidence=0.9,
                evidence_segment_ids=(UUID("44444444-4444-4444-8444-444444444444"),),
            ),
        ),
        needs_review=True,
    )
    with pytest.raises(ValidationError, match="неизвестный"):
        _snapshot(role_resolution=unknown_evidence)
