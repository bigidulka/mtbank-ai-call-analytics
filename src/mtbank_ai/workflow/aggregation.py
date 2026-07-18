"""Детерминированный fan-in четырёх agent outputs в публичный контракт."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID

from mtbank_ai.domain.agents import (
    ClassificationResult,
    ComplianceAssessment,
    ComplianceSeverity,
    QualityAssessment,
    SummaryResult,
)
from mtbank_ai.domain.analysis import (
    AnalysisMeta,
    AnalysisVersions,
    AnalyzeResponse,
    CompletedRunStatus,
    ComplianceView,
    GroundedActionItem,
    Grounding,
    PublicClassification,
    PublicComplianceIssue,
    PublicTranscriptSegment,
    QualityChecklist,
    QualityChecklistItem,
    QualityDetails,
    QualityScore,
    SanitizedAnalysisRecord,
    SanitizedComplianceIssue,
    SanitizedQualityChecklist,
)
from mtbank_ai.domain.transcript import TranscriptSnapshot
from mtbank_ai.policies import PolicyRegistry

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)")


class AggregationError(ValueError):
    """Нарушен trusted contract между terminal output и deterministic fan-in."""


@dataclass(frozen=True, slots=True)
class AggregatedAnalysis:
    response: AnalyzeResponse
    sanitized_record: SanitizedAnalysisRecord


def aggregate_analysis(
    transcript: TranscriptSnapshot,
    *,
    classification: ClassificationResult,
    quality: QualityAssessment,
    compliance: ComplianceAssessment,
    summary: SummaryResult,
    policies: PolicyRegistry,
    run_id: UUID,
    versions: AnalysisVersions,
    processing_ms: int,
) -> AggregatedAnalysis:
    """Валидирует references/enums и вычисляет derived values без участия LLM."""

    if processing_ms < 0:
        raise AggregationError("processing_ms не может быть отрицательным")
    known_segment_ids = {segment.id for segment in transcript.segments}
    _validate_classification(classification, policies, known_segment_ids)
    _validate_quality(quality, policies, known_segment_ids)
    _validate_compliance(compliance, policies, known_segment_ids)
    _validate_summary(summary, known_segment_ids)

    quality_details = QualityDetails(
        greeting=_quality_item(quality.greeting),
        need_detection=_quality_item(quality.need_detection),
        solution_provided=_quality_item(quality.solution_provided),
        farewell=_quality_item(quality.farewell),
    )
    checklist = QualityChecklist(
        greeting=quality.greeting.passed,
        need_detection=quality.need_detection.passed,
        solution_provided=quality.solution_provided.passed,
        farewell=quality.farewell.passed,
    )
    quality_total = _quality_total(quality, policies)
    compliance_issues = tuple(
        PublicComplianceIssue(
            rule_id=issue.rule_id,
            severity=issue.severity,
            evidence_segment_ids=issue.evidence_segment_ids,
            explanation=issue.explanation,
        )
        for issue in compliance.issues
    )
    action_items = tuple(item.text for item in summary.action_items)
    grounded_actions = tuple(
        GroundedActionItem(
            text=item.text,
            evidence_segment_ids=item.evidence_segment_ids,
        )
        for item in summary.action_items
    )
    needs_review = _needs_review(transcript, classification, quality, policies)
    response = AnalyzeResponse(
        transcript=tuple(
            PublicTranscriptSegment(
                id=segment.id,
                speaker=segment.speaker,
                start=segment.start,
                end=segment.end,
                text=segment.redacted_text,
            )
            for segment in transcript.segments
        ),
        classification=PublicClassification(
            topic=classification.topic,
            priority=classification.priority,
            confidence=classification.confidence,
            evidence_segment_ids=classification.evidence_segment_ids,
            rationale=classification.rationale,
            taxonomy_version=f"taxonomy/{policies.taxonomy.version}",
        ),
        quality_score=QualityScore(
            total=quality_total,
            checklist=checklist,
            details=quality_details,
            policy_version=f"quality/{policies.quality.version}",
        ),
        compliance=ComplianceView(
            passed=not any(issue.severity is ComplianceSeverity.BLOCKING for issue in compliance.issues),
            issues=compliance_issues,
            policy_version=f"compliance/{policies.compliance.version}",
        ),
        summary=summary.summary,
        action_items=action_items,
        grounding=Grounding(
            summary_evidence_segment_ids=summary.fact_segment_ids,
            action_items=grounded_actions,
        ),
        meta=AnalysisMeta(
            run_id=run_id,
            status=CompletedRunStatus.COMPLETED,
            versions=versions,
            processing_ms=processing_ms,
            needs_review=needs_review,
        ),
    )
    sanitized = SanitizedAnalysisRecord(
        run_id=run_id,
        classification_topic_id=classification.topic,
        classification_priority_id=classification.priority,
        classification_confidence=classification.confidence,
        classification_evidence_segment_ids=classification.evidence_segment_ids,
        quality_total=quality_total,
        quality_checklist=SanitizedQualityChecklist(
            greeting=quality.greeting.passed,
            need_detection=quality.need_detection.passed,
            solution_provided=quality.solution_provided.passed,
            farewell=quality.farewell.passed,
        ),
        quality_evidence_segment_ids=_unique_quality_evidence(quality),
        compliance_passed=response.compliance.passed,
        compliance_issues=tuple(
            SanitizedComplianceIssue(
                rule_id=issue.rule_id,
                severity=issue.severity,
                evidence_segment_ids=issue.evidence_segment_ids,
            )
            for issue in compliance.issues
        ),
        action_item_count=len(action_items),
        needs_review=needs_review,
        processing_ms=processing_ms,
        trusted_versions=versions,
    )
    return AggregatedAnalysis(response=response, sanitized_record=sanitized)


def _validate_classification(
    classification: ClassificationResult,
    policies: PolicyRegistry,
    known_segment_ids: set[UUID],
) -> None:
    try:
        allowed_priorities = policies.taxonomy.policy.allowed_priorities(classification.topic)
    except ValueError as error:
        raise AggregationError("classification topic не утверждён") from error
    if classification.priority not in allowed_priorities:
        raise AggregationError("classification priority не утверждён для topic")
    _require_known_evidence(classification.evidence_segment_ids, known_segment_ids)


def _validate_quality(quality: QualityAssessment, policies: PolicyRegistry, known_segment_ids: set[UUID]) -> None:
    for identifier, item in (
        ("greeting", quality.greeting),
        ("need_detection", quality.need_detection),
        ("solution_provided", quality.solution_provided),
        ("farewell", quality.farewell),
    ):
        policies.quality.policy.criterion(identifier)
        _require_known_evidence(item.evidence_segment_ids, known_segment_ids)


def _validate_compliance(
    compliance: ComplianceAssessment,
    policies: PolicyRegistry,
    known_segment_ids: set[UUID],
) -> None:
    rule_ids: set[str] = set()
    for issue in compliance.issues:
        if issue.rule_id in rule_ids:
            raise AggregationError("compliance issues не могут повторять rule")
        rule_ids.add(issue.rule_id)
        try:
            rule = policies.compliance.policy.rule(issue.rule_id)
        except ValueError as error:
            raise AggregationError("compliance rule не утверждён") from error
        if issue.severity is not rule.severity:
            raise AggregationError("compliance severity должна принадлежать policy rule")
        _require_known_evidence(issue.evidence_segment_ids, known_segment_ids)


def _validate_summary(summary: SummaryResult, known_segment_ids: set[UUID]) -> None:
    sentences = tuple(sentence for sentence in _SENTENCE_BOUNDARY.split(summary.summary.strip()) if sentence.strip())
    sentence_count = len(sentences)
    if not 3 <= sentence_count <= 5:
        raise AggregationError("summary должен содержать от трёх до пяти предложений")
    _require_known_evidence(summary.fact_segment_ids, known_segment_ids)
    for item in summary.action_items:
        _require_known_evidence(item.evidence_segment_ids, known_segment_ids)


def _require_known_evidence(evidence_ids: tuple[UUID, ...], known_segment_ids: set[UUID]) -> None:
    if not set(evidence_ids).issubset(known_segment_ids):
        raise AggregationError("evidence ID отсутствует в immutable transcript")


def _quality_item(item):  # type: ignore[no-untyped-def]
    return QualityChecklistItem(
        passed=item.passed,
        confidence=item.confidence,
        evidence_segment_ids=item.evidence_segment_ids,
        rationale=item.rationale,
    )


def _quality_total(quality: QualityAssessment, policies: PolicyRegistry) -> float:
    assessments = {
        "greeting": quality.greeting,
        "need_detection": quality.need_detection,
        "solution_provided": quality.solution_provided,
        "farewell": quality.farewell,
    }
    total = sum(
        (
            Decimal(str(criterion.weight)) * Decimal("100")
            for criterion in policies.quality.policy.criteria
            if assessments[criterion.id].passed
        ),
        Decimal("0"),
    )
    return float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _unique_quality_evidence(quality: QualityAssessment) -> tuple[UUID, ...]:
    values: list[UUID] = []
    for item in (quality.greeting, quality.need_detection, quality.solution_provided, quality.farewell):
        for segment_id in item.evidence_segment_ids:
            if segment_id not in values:
                values.append(segment_id)
    return tuple(values)


def _needs_review(
    transcript: TranscriptSnapshot,
    classification: ClassificationResult,
    quality: QualityAssessment,
    policies: PolicyRegistry,
) -> bool:
    quality_policy = policies.quality.policy
    agent_confidences = (
        classification.confidence,
        quality.greeting.confidence,
        quality.need_detection.confidence,
        quality.solution_provided.confidence,
        quality.farewell.confidence,
    )
    return (
        transcript.role_resolution.needs_review
        or any(segment.role_confidence < quality_policy.role_confidence_threshold for segment in transcript.segments)
        or any(confidence < quality_policy.review_confidence_threshold for confidence in agent_confidences)
    )
