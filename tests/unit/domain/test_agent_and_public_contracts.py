from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from mtbank_ai.domain.agents import ComplianceAssessment, QualityAssessment, QualityCriterionAssessment
from mtbank_ai.domain.analysis import (
    AnalysisMeta,
    AnalysisVersions,
    AnalyzeResponse,
    CompletedRunStatus,
    ComplianceView,
    GroundedActionItem,
    Grounding,
    PublicClassification,
    PublicTranscriptSegment,
    QualityChecklist,
    QualityChecklistItem,
    QualityDetails,
    QualityScore,
    SanitizedAnalysisRecord,
)
from mtbank_ai.domain.events import RunStatus
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import SpeakerRole

SEGMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
UNKNOWN_SEGMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
RUN_ID = UUID("33333333-3333-4333-8333-333333333333")


def _criterion() -> QualityCriterionAssessment:
    return QualityCriterionAssessment(
        passed=True,
        confidence=0.9,
        evidence_segment_ids=(SEGMENT_ID,),
        rationale="Критерий подтверждён транскриптом.",
    )


def _item(*, passed: bool = True) -> QualityChecklistItem:
    return QualityChecklistItem(
        passed=passed,
        confidence=0.9,
        evidence_segment_ids=(SEGMENT_ID,),
        rationale="Критерий подтверждён транскриптом.",
    )


def _response(**changes: object) -> AnalyzeResponse:
    values: dict[str, object] = {
        "transcript": (
            PublicTranscriptSegment(
                id=SEGMENT_ID,
                speaker=SpeakerRole.OPERATOR,
                start=0.0,
                end=1.0,
                text="Добрый день.",
            ),
        ),
        "classification": PublicClassification(
            topic="кредиты",
            priority="medium",
            confidence=0.95,
            evidence_segment_ids=(SEGMENT_ID,),
            rationale="Тема подтверждена репликой клиента.",
            taxonomy_version="taxonomy/v1",
        ),
        "quality_score": QualityScore(
            total=75.0,
            checklist=QualityChecklist(
                greeting=True,
                need_detection=True,
                solution_provided=True,
                farewell=False,
            ),
            details=QualityDetails(
                greeting=_item(),
                need_detection=_item(),
                solution_provided=_item(),
                farewell=_item(passed=False),
            ),
            policy_version="quality/v1",
        ),
        "compliance": ComplianceView(passed=True, issues=(), policy_version="compliance/v1"),
        "summary": "Клиент запросил информацию о кредите.",
        "action_items": ("Отправить условия.",),
        "grounding": Grounding(
            summary_evidence_segment_ids=(SEGMENT_ID,),
            action_items=(GroundedActionItem(text="Отправить условия.", evidence_segment_ids=(SEGMENT_ID,)),),
        ),
        "meta": AnalysisMeta(
            run_id=RUN_ID,
            status=CompletedRunStatus.COMPLETED,
            versions=AnalysisVersions(
                code_sha="abcdef0",
                prompt_bundle_hash="a" * 64,
                taxonomy_version="taxonomy/v1",
                quality_rubric_version="quality/v1",
                compliance_policy_version="compliance/v1",
                asr=ComponentRevision(
                    package="faster-whisper",
                    package_version="1.0.0",
                    model_id="large-v3",
                    model_revision="asr/v1",
                ),
                alignment=ComponentRevision(
                    package="whisperx",
                    package_version="3.0.0",
                    model_id="wav2vec2",
                    model_revision="alignment/v1",
                ),
                diarization=ComponentRevision(
                    package="pyannote.audio",
                    package_version="3.3.0",
                    model_id="speaker-diarization",
                    model_revision="diarization/v1",
                ),
            ),
            processing_ms=125,
            needs_review=False,
        ),
    }
    values.update(changes)
    return AnalyzeResponse.model_validate(values)


def test_internal_agent_dtos_reject_derived_total_and_passed() -> None:
    quality = {
        "greeting": _criterion(),
        "need_detection": _criterion(),
        "solution_provided": _criterion(),
        "farewell": _criterion(),
        "total": 100,
    }
    with pytest.raises(ValidationError, match="total"):
        QualityAssessment.model_validate(quality)
    with pytest.raises(ValidationError, match="passed"):
        ComplianceAssessment.model_validate({"issues": (), "passed": True})


def test_public_json_contract_exposes_assignment_fields_and_additive_grounding() -> None:
    payload = _response().model_dump(mode="json")

    assert set(payload) == {
        "transcript",
        "classification",
        "quality_score",
        "compliance",
        "summary",
        "action_items",
        "grounding",
        "meta",
    }
    assert set(payload["transcript"][0]) == {"id", "speaker", "start", "end", "text"}
    assert payload["quality_score"]["checklist"] == {
        "greeting": True,
        "need_detection": True,
        "solution_provided": True,
        "farewell": False,
    }
    assert set(payload["quality_score"]["details"]) == {
        "greeting",
        "need_detection",
        "solution_provided",
        "farewell",
    }
    assert set(payload["classification"]) == {
        "topic",
        "priority",
        "confidence",
        "evidence_segment_ids",
        "rationale",
        "taxonomy_version",
    }
    assert AnalyzeResponse.model_validate_json(_response().model_dump_json()) == _response()


def test_public_contract_rejects_string_bool_and_numeric_coercion() -> None:
    payload = _response().model_dump(mode="json")
    payload["quality_score"]["checklist"]["greeting"] = "false"
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["quality_score"]["total"] = "75.0"
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["quality_score"]["details"]["greeting"]["passed"] = False
    with pytest.raises(ValidationError, match="checklist"):
        AnalyzeResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["classification"]["confidence"] = "0.95"
    payload["meta"]["processing_ms"] = "125"
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)


def test_public_contract_rejects_non_completed_status_unknown_evidence_and_version_drift() -> None:
    payload = _response().model_dump(mode="json")
    payload["meta"]["status"] = RunStatus.PROCESSING
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["classification"]["evidence_segment_ids"] = [str(UNKNOWN_SEGMENT_ID)]
    with pytest.raises(ValidationError, match="существовать"):
        AnalyzeResponse.model_validate(payload)

    payload = _response().model_dump(mode="json")
    payload["classification"]["taxonomy_version"] = "taxonomy/v2"
    with pytest.raises(ValidationError, match="taxonomy version"):
        AnalyzeResponse.model_validate(payload)


def test_full_public_response_cannot_validate_as_sanitized_record() -> None:
    with pytest.raises(ValidationError):
        SanitizedAnalysisRecord.model_validate(_response().model_dump())


def test_public_models_are_frozen_and_forbid_extra_input() -> None:
    response = _response()
    with pytest.raises(ValidationError, match="frozen"):
        response.summary = "Новая сводка"
    with pytest.raises(ValidationError, match="Extra inputs"):
        PublicClassification.model_validate({**response.classification.model_dump(), "extra": "forbidden"})
