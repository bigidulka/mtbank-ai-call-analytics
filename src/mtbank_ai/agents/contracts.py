"""Tool DTO для core agents; terminal inputs не допускают derived business fields."""

from __future__ import annotations

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from mtbank_ai.domain.agents import ClassificationResult
from mtbank_ai.domain.base import Confidence, LongText, NonEmptyId, NonNegativeFloat, NonNegativeInt, StrictFrozenModel
from mtbank_ai.domain.transcript import SpeakerRole


class EmptyToolInput(StrictFrozenModel):
    pass


class TranscriptGetInput(StrictFrozenModel):
    segment_ids: Annotated[tuple[UUID, ...], Field(min_length=1, max_length=20)]

    @model_validator(mode="after")
    def require_unique_ids(self) -> Self:
        if len(set(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("segment IDs должны быть уникальны")
        return self


class TranscriptSearchInput(StrictFrozenModel):
    query: LongText
    limit: Annotated[int, Field(ge=1, le=20)] = 8


class AgentTranscriptSegment(StrictFrozenModel):
    id: UUID
    speaker: SpeakerRole
    start: NonNegativeFloat
    end: Annotated[float, Field(gt=0.0)]
    redacted_text: LongText

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start >= self.end:
            raise ValueError("segment start должен быть меньше end")
        return self


class TranscriptSegmentsOutput(StrictFrozenModel):
    segments: tuple[AgentTranscriptSegment, ...]


class TranscriptStatisticsOutput(StrictFrozenModel):
    segment_count: NonNegativeInt
    duration_seconds: Annotated[float, Field(gt=0.0)]
    operator_segment_count: NonNegativeInt
    client_segment_count: NonNegativeInt


class TaxonomyTopicOutput(StrictFrozenModel):
    id: NonEmptyId
    description: LongText
    allowed_priorities: tuple[NonEmptyId, ...]


class TaxonomyOutput(StrictFrozenModel):
    version: NonEmptyId
    owner: NonEmptyId
    effective_date: NonEmptyId
    topics: tuple[TaxonomyTopicOutput, ...]


class QualityCriterionOutput(StrictFrozenModel):
    id: NonEmptyId
    weight: Annotated[float, Field(gt=0.0, le=1.0)]
    description: LongText


class QualityRubricOutput(StrictFrozenModel):
    version: NonEmptyId
    owner: NonEmptyId
    effective_date: NonEmptyId
    criteria: tuple[QualityCriterionOutput, ...]


class ComplianceRuleGetInput(StrictFrozenModel):
    rule_id: NonEmptyId


class ComplianceRuleOutput(StrictFrozenModel):
    rule_id: NonEmptyId
    severity: Literal["info", "warning", "blocking"]
    description: LongText


class ComplianceRuleListOutput(StrictFrozenModel):
    version: NonEmptyId
    rules: tuple[ComplianceRuleOutput, ...]


class ClassificationSubmission(StrictFrozenModel):
    topic: Literal["кредиты", "карты", "переводы", "жалобы", "другое"]
    priority: Literal["low", "medium", "high"]
    confidence: Confidence
    evidence_segment_ids: Annotated[tuple[UUID, ...], Field(min_length=1)]
    rationale: LongText

    @model_validator(mode="after")
    def require_unique_evidence(self) -> Self:
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise ValueError("evidence segment IDs должны быть уникальны")
        return self

    def to_result(self) -> ClassificationResult:
        return ClassificationResult.model_validate(self.model_dump(), strict=True)
