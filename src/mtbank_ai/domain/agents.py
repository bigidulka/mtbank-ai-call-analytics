"""Строгие внутренние результаты четырёх обязательных агентов."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, model_validator

from mtbank_ai.domain.base import Confidence, LongText, NonEmptyId, StrictFrozenModel

EvidenceSegmentIds = Annotated[tuple[UUID, ...], Field(min_length=1)]


def _require_unique(values: tuple[UUID, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("evidence segment IDs должны быть уникальны")


class ClassificationResult(StrictFrozenModel):
    topic: NonEmptyId
    priority: NonEmptyId
    confidence: Confidence
    evidence_segment_ids: EvidenceSegmentIds
    rationale: LongText

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class QualityCriterionAssessment(StrictFrozenModel):
    passed: bool
    confidence: Confidence
    evidence_segment_ids: EvidenceSegmentIds
    rationale: LongText

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class QualityAssessment(StrictFrozenModel):
    greeting: QualityCriterionAssessment
    need_detection: QualityCriterionAssessment
    solution_provided: QualityCriterionAssessment
    farewell: QualityCriterionAssessment


class ComplianceSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


class ComplianceIssue(StrictFrozenModel):
    rule_id: NonEmptyId
    severity: ComplianceSeverity
    evidence_segment_ids: EvidenceSegmentIds
    explanation: LongText

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class ComplianceAssessment(StrictFrozenModel):
    issues: tuple[ComplianceIssue, ...]


class ActionItem(StrictFrozenModel):
    text: LongText
    evidence_segment_ids: EvidenceSegmentIds

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class SummaryResult(StrictFrozenModel):
    summary: LongText
    fact_segment_ids: EvidenceSegmentIds
    action_items: tuple[ActionItem, ...]

    @model_validator(mode="after")
    def validate_facts(self) -> Self:
        _require_unique(self.fact_segment_ids)
        return self
