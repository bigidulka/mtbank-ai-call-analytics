"""Публичный JSON-контракт результата анализа."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, Strict, StrictBool, model_validator

from mtbank_ai.domain.agents import ComplianceSeverity
from mtbank_ai.domain.base import (
    Confidence,
    FrozenModel,
    LongText,
    NonEmptyId,
    NonNegativeInt,
    Sha256,
    StrictFrozenModel,
)
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import SpeakerRole

EvidenceSegmentIds = Annotated[tuple[UUID, ...], Field(min_length=1)]


def _require_unique(values: tuple[UUID, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("evidence segment IDs должны быть уникальны")


class PublicTranscriptSegment(FrozenModel):
    id: UUID
    speaker: SpeakerRole
    start: Annotated[float, Strict(), Field(ge=0.0)]
    end: Annotated[float, Strict(), Field(gt=0.0)]
    text: LongText

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start >= self.end:
            raise ValueError("начало публичного сегмента должно быть раньше конца")
        return self


class PublicClassification(FrozenModel):
    topic: NonEmptyId
    priority: NonEmptyId
    confidence: Confidence
    evidence_segment_ids: EvidenceSegmentIds
    rationale: LongText
    taxonomy_version: NonEmptyId

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class QualityChecklistItem(FrozenModel):
    passed: StrictBool
    confidence: Confidence
    evidence_segment_ids: EvidenceSegmentIds
    rationale: LongText

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class QualityChecklist(FrozenModel):
    greeting: StrictBool
    need_detection: StrictBool
    solution_provided: StrictBool
    farewell: StrictBool


class QualityDetails(FrozenModel):
    greeting: QualityChecklistItem
    need_detection: QualityChecklistItem
    solution_provided: QualityChecklistItem
    farewell: QualityChecklistItem


class QualityScore(FrozenModel):
    total: Annotated[float, Strict(), Field(ge=0.0, le=100.0)]
    checklist: QualityChecklist
    details: QualityDetails
    policy_version: NonEmptyId

    @model_validator(mode="after")
    def validate_checklist_matches_details(self) -> Self:
        for criterion in ("greeting", "need_detection", "solution_provided", "farewell"):
            if getattr(self.checklist, criterion) != getattr(self.details, criterion).passed:
                raise ValueError("quality checklist должен совпадать с details.passed")
        return self


class PublicComplianceIssue(FrozenModel):
    rule_id: NonEmptyId
    severity: ComplianceSeverity
    evidence_segment_ids: EvidenceSegmentIds
    explanation: LongText

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class ComplianceView(FrozenModel):
    passed: StrictBool
    issues: tuple[PublicComplianceIssue, ...]
    policy_version: NonEmptyId


class GroundedActionItem(FrozenModel):
    text: LongText
    evidence_segment_ids: EvidenceSegmentIds

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class Grounding(FrozenModel):
    summary_evidence_segment_ids: EvidenceSegmentIds
    action_items: tuple[GroundedActionItem, ...]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.summary_evidence_segment_ids)
        return self


class AnalysisVersions(FrozenModel):
    schema_version: Literal["1"] = "1"
    code_sha: NonEmptyId
    prompt_bundle_hash: Sha256
    taxonomy_version: NonEmptyId
    quality_rubric_version: NonEmptyId
    compliance_policy_version: NonEmptyId
    asr: ComponentRevision
    alignment: ComponentRevision
    diarization: ComponentRevision


class CompletedRunStatus(StrEnum):
    COMPLETED = "completed"


class AnalysisMeta(FrozenModel):
    run_id: UUID
    status: CompletedRunStatus
    versions: AnalysisVersions
    processing_ms: Annotated[int, Strict(), Field(ge=0)]
    needs_review: StrictBool


class AnalyzeResponse(FrozenModel):
    transcript: tuple[PublicTranscriptSegment, ...]
    classification: PublicClassification
    quality_score: QualityScore
    compliance: ComplianceView
    summary: LongText
    action_items: tuple[LongText, ...]
    grounding: Grounding
    meta: AnalysisMeta

    @model_validator(mode="after")
    def validate_public_contract(self) -> Self:
        segment_ids = {segment.id for segment in self.transcript}
        if len(segment_ids) != len(self.transcript):
            raise ValueError("публичные transcript segment IDs должны быть уникальны")

        evidence_ids = set(self.classification.evidence_segment_ids)
        for item in (
            self.quality_score.details.greeting,
            self.quality_score.details.need_detection,
            self.quality_score.details.solution_provided,
            self.quality_score.details.farewell,
        ):
            evidence_ids.update(item.evidence_segment_ids)
        for issue in self.compliance.issues:
            evidence_ids.update(issue.evidence_segment_ids)
        evidence_ids.update(self.grounding.summary_evidence_segment_ids)
        for item in self.grounding.action_items:
            evidence_ids.update(item.evidence_segment_ids)
        if not evidence_ids.issubset(segment_ids):
            raise ValueError("evidence segment IDs должны существовать в transcript")

        versions = self.meta.versions
        if self.classification.taxonomy_version != versions.taxonomy_version:
            raise ValueError("taxonomy version должна совпадать с meta")
        if self.quality_score.policy_version != versions.quality_rubric_version:
            raise ValueError("quality rubric version должна совпадать с meta")
        if self.compliance.policy_version != versions.compliance_policy_version:
            raise ValueError("compliance policy version должна совпадать с meta")
        if tuple(item.text for item in self.grounding.action_items) != self.action_items:
            raise ValueError("grounding action items должны совпадать с action_items")
        return self


class SanitizedQualityChecklist(StrictFrozenModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool


class SanitizedComplianceIssue(StrictFrozenModel):
    rule_id: NonEmptyId
    severity: ComplianceSeverity
    evidence_segment_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.evidence_segment_ids)
        return self


class SanitizedAnalysisRecord(StrictFrozenModel):
    """Единственный разрешённый к хранению срез результата анализа."""

    schema_version: Literal["1"] = "1"
    run_id: UUID
    classification_topic_id: NonEmptyId
    classification_priority_id: NonEmptyId
    classification_confidence: Confidence
    classification_evidence_segment_ids: tuple[UUID, ...] = ()
    quality_total: Annotated[float, Strict(), Field(ge=0.0, le=100.0)]
    quality_checklist: SanitizedQualityChecklist
    quality_evidence_segment_ids: tuple[UUID, ...] = ()
    compliance_passed: bool
    compliance_issues: tuple[SanitizedComplianceIssue, ...]
    action_item_count: NonNegativeInt
    needs_review: bool
    processing_ms: NonNegativeInt
    trusted_versions: AnalysisVersions

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_unique(self.classification_evidence_segment_ids)
        _require_unique(self.quality_evidence_segment_ids)
        return self
