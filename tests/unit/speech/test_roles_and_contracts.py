from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from mtbank_ai.domain.transcript import (
    RoleAssignment,
    RoleResolution,
    RoleResolutionSource,
    SpeakerRole,
    TranscriptSegment,
)
from mtbank_ai.speech.contracts import (
    DiarizedSegment,
    ResolvedRole,
    RoleResolutionCandidate,
    RoleResolutionDecision,
    SpeakerRoleMapping,
)
from mtbank_ai.speech.roles import RoleResolutionRequiredError, resolve_roles

SEGMENT_ID = UUID("11111111-1111-4111-8111-111111111111")


def _segment() -> DiarizedSegment:
    return DiarizedSegment(
        id=SEGMENT_ID,
        original_speaker_id="SPEAKER_00",
        speaker_confidence=0.8,
        start=0.0,
        end=1.0,
        text="Добрый день.",
    )


class Resolver:
    def __init__(self, *, confidence: float = 0.7) -> None:
        self.candidates: tuple[RoleResolutionCandidate, ...] = ()
        self.confidence = confidence

    def resolve(self, candidates: tuple[RoleResolutionCandidate, ...]) -> RoleResolutionDecision:
        self.candidates = candidates
        return RoleResolutionDecision(
            roles=tuple(
                ResolvedRole(
                    original_speaker_id=candidate.original_speaker_id,
                    role=SpeakerRole.CLIENT,
                    confidence=self.confidence,
                    evidence="resolver/v1",
                )
                for candidate in candidates
            )
        )


def test_metadata_mapping_has_priority_and_resolver_never_sees_mapped_speaker() -> None:
    resolver = Resolver()

    resolution = resolve_roles(
        (_segment(),),
        metadata_mappings=(
            SpeakerRoleMapping(
                original_speaker_id="SPEAKER_00",
                role=SpeakerRole.OPERATOR,
                confidence=1.0,
                evidence="trusted-call-metadata",
            ),
        ),
        resolver=resolver,
    )

    assignment = resolution.assignments[0]
    assert resolver.candidates == ()
    assert assignment.role is SpeakerRole.OPERATOR
    assert assignment.source is RoleResolutionSource.METADATA
    assert assignment.resolution_evidence == "trusted-call-metadata"
    assert not resolution.needs_review


def test_unresolved_role_stops_before_snapshot_and_never_uses_first_speaker_fallback() -> None:
    with pytest.raises(RoleResolutionRequiredError) as error:
        resolve_roles((_segment(),))

    candidate = error.value.candidates[0]
    assert candidate.original_speaker_id == "SPEAKER_00"
    assert candidate.speaker_confidence == 0.8
    assert candidate.evidence_segment_ids == (SEGMENT_ID,)


def test_complete_low_confidence_resolver_assignment_keeps_exact_role_and_marks_review() -> None:
    resolution = resolve_roles((_segment(),), resolver=Resolver(confidence=0.7), review_confidence_threshold=0.75)

    assert resolution.assignments[0].role is SpeakerRole.CLIENT
    assert resolution.assignments[0].source is RoleResolutionSource.RESOLVER
    assert resolution.needs_review


def test_existing_transcript_domain_rejects_unresolved_role_fields() -> None:
    with pytest.raises(ValidationError):
        RoleAssignment.model_validate(
            {
                "original_speaker_id": "SPEAKER_00",
                "role": None,
                "confidence": 0.0,
                "evidence_segment_ids": (SEGMENT_ID,),
            }
        )
    with pytest.raises(ValidationError):
        TranscriptSegment.model_validate(
            {
                "id": SEGMENT_ID,
                "original_speaker_id": "SPEAKER_00",
                "speaker": None,
                "role_confidence": 0.0,
                "start": 0.0,
                "end": 1.0,
                "text": "Реплика.",
                "redacted_text": "Реплика.",
            }
        )
    assert {role.value for role in SpeakerRole} == {"Оператор", "Клиент"}


def test_policy_assignment_requires_exact_policy_provenance() -> None:
    policy_assignment = RoleAssignment(
        original_speaker_id="SPEAKER_00",
        role=SpeakerRole.OPERATOR,
        confidence=0.9,
        evidence_segment_ids=(SEGMENT_ID,),
        source=RoleResolutionSource.POLICY,
        resolution_evidence="v1/test",
    )

    with pytest.raises(ValidationError, match="policy provenance"):
        RoleResolution(assignments=(policy_assignment,), needs_review=False)


def test_two_speaker_resolution_requires_distinct_public_roles() -> None:
    assignments = (
        RoleAssignment(
            original_speaker_id="SPEAKER_00",
            role=SpeakerRole.OPERATOR,
            confidence=0.9,
            evidence_segment_ids=(SEGMENT_ID,),
        ),
        RoleAssignment(
            original_speaker_id="SPEAKER_01",
            role=SpeakerRole.OPERATOR,
            confidence=0.9,
            evidence_segment_ids=(UUID("22222222-2222-4222-8222-222222222222"),),
        ),
    )

    with pytest.raises(ValidationError, match="two-speaker"):
        RoleResolution(assignments=assignments, needs_review=False)
