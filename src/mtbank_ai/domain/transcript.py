"""Неизменяемый транскрипт после распознавания и разрешения ролей."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from mtbank_ai.domain.base import (
    Confidence,
    LongText,
    NonEmptyId,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    Sha256,
    StrictFrozenModel,
    UtcDateTime,
)
from mtbank_ai.domain.provenance import ComponentRevision


class SpeakerRole(StrEnum):
    OPERATOR = "Оператор"
    CLIENT = "Клиент"


class RoleResolutionSource(StrEnum):
    """Источник явного назначения роли без эвристики порядка спикеров."""

    LEGACY = "legacy"
    METADATA = "metadata"
    RESOLVER = "resolver"
    POLICY = "policy"


class RolePolicyProvenance(StrictFrozenModel):
    policy_id: NonEmptyId
    version: NonEmptyId
    owner: NonEmptyId
    effective_date: NonEmptyId
    sha256: Sha256

    @field_validator("effective_date")
    @classmethod
    def require_iso_date(cls, value: str) -> str:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("role policy effective_date должен быть ISO-8601 датой") from error
        if parsed.isoformat() != value:
            raise ValueError("role policy effective_date должен быть canonical ISO-8601 датой")
        return value


class WordTimestamp(StrictFrozenModel):
    word: NonEmptyId
    start: NonNegativeFloat
    end: PositiveFloat
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало слова должно быть раньше конца")
        return self


class RoleAssignment(StrictFrozenModel):
    original_speaker_id: NonEmptyId
    role: SpeakerRole
    confidence: Confidence
    evidence_segment_ids: Annotated[tuple[UUID, ...], Field(min_length=1)]
    source: RoleResolutionSource = RoleResolutionSource.LEGACY
    resolution_evidence: NonEmptyId = "legacy"

    @model_validator(mode="after")
    def require_unique_evidence(self) -> Self:
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise ValueError("evidence segment IDs должны быть уникальны")
        return self


class RoleResolution(StrictFrozenModel):
    assignments: Annotated[tuple[RoleAssignment, ...], Field(min_length=1)]
    needs_review: bool
    policy_provenance: RolePolicyProvenance | None = None

    @model_validator(mode="after")
    def require_unique_speakers(self) -> Self:
        speaker_ids = tuple(item.original_speaker_id for item in self.assignments)
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ValueError("original speaker IDs должны быть уникальны")
        has_policy_assignment = any(item.source is RoleResolutionSource.POLICY for item in self.assignments)
        if has_policy_assignment != (self.policy_provenance is not None):
            raise ValueError("policy role assignments требуют точную policy provenance")
        if len(self.assignments) == 2 and {item.role for item in self.assignments} != {
            SpeakerRole.OPERATOR,
            SpeakerRole.CLIENT,
        }:
            raise ValueError("two-speaker role resolution требует Оператор и Клиент")
        return self


class ASRProviderMetadata(StrictFrozenModel):
    """Sanitized provenance returned by the canonical remote ASR provider."""

    provider: NonEmptyId
    model: NonEmptyId
    endpoint_fingerprint: Sha256
    request_id: NonEmptyId | None = None
    usage_seconds: NonNegativeFloat | None = None


class ASRMetadata(StrictFrozenModel):
    asr: ComponentRevision
    alignment: ComponentRevision
    diarization: ComponentRevision
    language: NonEmptyId
    processing_ms: NonNegativeInt
    provider: ASRProviderMetadata | None = None


class TranscriptSegment(StrictFrozenModel):
    id: UUID
    original_speaker_id: NonEmptyId
    speaker: SpeakerRole
    role_confidence: Confidence
    speaker_confidence: Confidence | None = None
    start: NonNegativeFloat
    end: PositiveFloat
    text: LongText
    redacted_text: LongText
    word_timestamps: tuple[WordTimestamp, ...] = ()

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало сегмента должно быть раньше конца")

        previous_start = self.start
        previous_end = self.start
        for word in self.word_timestamps:
            if word.start < self.start or word.end > self.end:
                raise ValueError("временная метка слова выходит за границы сегмента")
            if word.start < previous_start or word.end < previous_end:
                raise ValueError("временные метки слов должны быть монотонны")
            previous_start = word.start
            previous_end = word.end
        return self


class TranscriptSnapshot(StrictFrozenModel):
    transcript_id: UUID
    audio_sha256: Sha256
    revision: NonEmptyId
    language: NonEmptyId
    duration_seconds: PositiveFloat
    segments: Annotated[tuple[TranscriptSegment, ...], Field(min_length=1)]
    role_resolution: RoleResolution
    asr_metadata: ASRMetadata
    created_at: UtcDateTime

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        segment_ids = tuple(segment.id for segment in self.segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segment IDs должны быть уникальны")

        previous_start = -1.0
        previous_end = 0.0
        for segment in self.segments:
            if segment.start < previous_start:
                raise ValueError("сегменты должны быть отсортированы по start")
            if segment.end < previous_end:
                raise ValueError("сегменты должны иметь монотонные timestamps")
            if segment.end > self.duration_seconds:
                raise ValueError("сегмент выходит за duration транскрипта")
            previous_start = segment.start
            previous_end = segment.end

        assignments = {
            assignment.original_speaker_id: assignment
            for assignment in self.role_resolution.assignments
        }
        segment_speaker_ids = {segment.original_speaker_id for segment in self.segments}
        if set(assignments) != segment_speaker_ids:
            raise ValueError("role resolution должен покрывать все original speaker IDs")

        known_segment_ids = set(segment_ids)
        for assignment in self.role_resolution.assignments:
            if not set(assignment.evidence_segment_ids).issubset(known_segment_ids):
                raise ValueError("role resolution ссылается на неизвестный сегмент")
        for segment in self.segments:
            assignment = assignments[segment.original_speaker_id]
            if assignment.role != segment.speaker or assignment.confidence != segment.role_confidence:
                raise ValueError("роль сегмента не совпадает с role resolution")
        if self.asr_metadata.language != self.language:
            raise ValueError("язык ASR metadata не совпадает с языком транскрипта")
        return self
