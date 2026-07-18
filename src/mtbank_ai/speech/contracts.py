"""Строгие transport и pipeline DTO для канонической batch-расшифровки."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from mtbank_ai.domain.base import (
    Confidence,
    FrozenModel,
    LongText,
    NonEmptyId,
    NonNegativeFloat,
    PositiveFloat,
    StrictFrozenModel,
)
from mtbank_ai.domain.transcript import (
    ASRProviderMetadata,
    RolePolicyProvenance,
    SpeakerRole,
    TranscriptSnapshot,
    WordTimestamp,
)


class SpeakerRoleMapping(FrozenModel):
    """Явное сопоставление diarization label с публичной ролью."""

    original_speaker_id: NonEmptyId
    role: SpeakerRole
    confidence: Confidence = 1.0
    evidence: NonEmptyId = "caller_metadata"


class SpeechMetadata(StrictFrozenModel):
    """Неаудио metadata, разрешённая на входе внутреннего speech API."""

    role_mappings: tuple[SpeakerRoleMapping, ...] = ()

    @field_validator("role_mappings", mode="before")
    @classmethod
    def parse_json_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def require_unique_mapping_speakers(self) -> Self:
        speaker_ids = tuple(mapping.original_speaker_id for mapping in self.role_mappings)
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ValueError("role mappings должны содержать уникальные original speaker IDs")
        return self


@dataclass(frozen=True, slots=True)
class SpeechFile:
    """Байты входного файла вне JSON/domain persistence boundary."""

    filename: str
    content_type: str
    content: bytes = field(repr=False)
    metadata: SpeechMetadata = field(default_factory=SpeechMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename обязателен")
        if not isinstance(self.content_type, str) or not self.content_type.strip():
            raise ValueError("content_type обязателен")
        if not isinstance(self.content, bytes):
            raise TypeError("content должен быть bytes")


class RecognizedWord(StrictFrozenModel):
    text: NonEmptyId
    start: NonNegativeFloat
    end: PositiveFloat
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало распознанного слова должно быть раньше конца")
        return self


class RecognizedSegment(StrictFrozenModel):
    start: NonNegativeFloat
    end: PositiveFloat
    text: LongText
    words: tuple[RecognizedWord, ...] = ()

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало распознанного сегмента должно быть раньше конца")
        previous_start = self.start
        for word in self.words:
            if word.start < self.start or word.end > self.end:
                raise ValueError("слово выходит за границы распознанного сегмента")
            if word.start < previous_start:
                raise ValueError("слова распознанного сегмента должны быть монотонны")
            previous_start = word.start
        return self


class TranscriptionResult(StrictFrozenModel):
    language: NonEmptyId
    segments: tuple[RecognizedSegment, ...]
    provider_metadata: ASRProviderMetadata | None = None

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        starts = tuple(segment.start for segment in self.segments)
        if starts != tuple(sorted(starts)):
            raise ValueError("распознанные сегменты должны быть отсортированы по start")
        return self


class AlignedSegment(StrictFrozenModel):
    start: NonNegativeFloat
    end: PositiveFloat
    text: LongText
    words: tuple[WordTimestamp, ...] = ()

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало aligned сегмента должно быть раньше конца")
        previous_start = self.start
        for word in self.words:
            if word.start < self.start or word.end > self.end:
                raise ValueError("слово выходит за границы aligned сегмента")
            if word.start < previous_start:
                raise ValueError("слова aligned сегмента должны быть монотонны")
            previous_start = word.start
        return self


class DiarizationTurn(StrictFrozenModel):
    original_speaker_id: NonEmptyId
    start: NonNegativeFloat
    end: PositiveFloat
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало diarization turn должно быть раньше конца")
        return self


class SpeakerAttributedSegment(StrictFrozenModel):
    start: NonNegativeFloat
    end: PositiveFloat
    text: LongText
    words: tuple[WordTimestamp, ...] = ()
    original_speaker_id: NonEmptyId | None = None
    speaker_confidence: Confidence | None = None

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало speaker-attributed сегмента должно быть раньше конца")
        previous_start = self.start
        for word in self.words:
            if word.start < self.start or word.end > self.end:
                raise ValueError("слово выходит за границы speaker-attributed сегмента")
            if word.start < previous_start:
                raise ValueError("слова speaker-attributed сегмента должны быть монотонны")
            previous_start = word.start
        return self


class DiarizedSegment(StrictFrozenModel):
    """Internal pre-resolution segment; it intentionally has no public role field."""

    id: UUID
    original_speaker_id: NonEmptyId
    speaker_confidence: Confidence | None = None
    start: NonNegativeFloat
    end: PositiveFloat
    text: LongText
    word_timestamps: tuple[WordTimestamp, ...] = ()

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало diarized сегмента должно быть раньше конца")
        previous_start = self.start
        previous_end = self.start
        for word in self.word_timestamps:
            if word.start < self.start or word.end > self.end:
                raise ValueError("слово выходит за границы diarized сегмента")
            if word.start < previous_start or word.end < previous_end:
                raise ValueError("слова diarized сегмента должны быть монотонны")
            previous_start = word.start
            previous_end = word.end
        return self


class RoleSegmentEvidence(StrictFrozenModel):
    segment_id: UUID
    text: LongText


class RoleResolutionCandidate(StrictFrozenModel):
    original_speaker_id: NonEmptyId
    evidence_segment_ids: Annotated[tuple[UUID, ...], Field(min_length=1)]
    evidence_segments: tuple[RoleSegmentEvidence, ...] = ()
    speaker_confidence: Confidence | None = None

    @model_validator(mode="after")
    def require_unique_evidence(self) -> Self:
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise ValueError("evidence segment IDs должны быть уникальны")
        evidence_segment_ids = tuple(item.segment_id for item in self.evidence_segments)
        if self.evidence_segments and evidence_segment_ids != self.evidence_segment_ids:
            raise ValueError("role evidence segments должны точно соответствовать evidence segment IDs")
        return self


class ResolvedRole(StrictFrozenModel):
    original_speaker_id: NonEmptyId
    role: SpeakerRole
    confidence: Confidence
    evidence: NonEmptyId
    evidence_segment_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def require_unique_evidence(self) -> Self:
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise ValueError("resolved role evidence segment IDs должны быть уникальны")
        return self


class RoleResolutionDecision(StrictFrozenModel):
    roles: tuple[ResolvedRole, ...] = ()
    policy_provenance: RolePolicyProvenance | None = None


class SpeechTranscriptionResponse(StrictFrozenModel):
    """Ответ internal-only `/v1/transcribe`; public API не сериализует unresolved roles."""

    transcript: TranscriptSnapshot
