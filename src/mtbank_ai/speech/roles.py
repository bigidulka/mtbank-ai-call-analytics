"""Explicit metadata or bounded-agent role resolution without speaker-order heuristics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol
from uuid import UUID

from mtbank_ai.domain.transcript import RoleAssignment, RoleResolution, RoleResolutionSource, SpeakerRole
from mtbank_ai.speech.contracts import (
    DiarizedSegment,
    ResolvedRole,
    RoleResolutionCandidate,
    RoleResolutionDecision,
    RoleSegmentEvidence,
    SpeakerRoleMapping,
)


class RoleResolverPort(Protocol):
    def resolve(self, candidates: tuple[RoleResolutionCandidate, ...]) -> RoleResolutionDecision: ...


class RoleResolutionRequiredError(Exception):
    """Trusted metadata or role agent did not provide a complete grounded assignment."""

    def __init__(self, candidates: tuple[RoleResolutionCandidate, ...]) -> None:
        self.candidates = candidates
        super().__init__("role_resolution_required")


def resolve_roles(
    segments: tuple[DiarizedSegment, ...],
    *,
    metadata_mappings: tuple[SpeakerRoleMapping, ...] = (),
    resolver: RoleResolverPort | None = None,
    review_confidence_threshold: float = 0.75,
) -> RoleResolution:
    candidates = _candidates(segments)
    mappings = {mapping.original_speaker_id: mapping for mapping in metadata_mappings}
    assignments: dict[str, RoleAssignment] = {}
    unresolved: list[RoleResolutionCandidate] = []

    for candidate in candidates:
        mapping = mappings.get(candidate.original_speaker_id)
        if mapping is None:
            unresolved.append(candidate)
            continue
        assignments[candidate.original_speaker_id] = RoleAssignment(
            original_speaker_id=candidate.original_speaker_id,
            role=mapping.role,
            confidence=mapping.confidence,
            evidence_segment_ids=candidate.evidence_segment_ids,
            source=RoleResolutionSource.METADATA,
            resolution_evidence=mapping.evidence,
        )

    prompt_provenance = None
    if unresolved and resolver is not None:
        decision = resolver.resolve(tuple(unresolved))
        if not isinstance(decision, RoleResolutionDecision):
            raise ValueError("role resolver должен вернуть typed role resolution decision")
        _validate_resolver_output(unresolved, decision.roles)
        prompt_provenance = decision.agent_provenance
        if decision.roles and prompt_provenance is None:
            raise ValueError("role agent output требует prompt provenance")
        candidates_by_id = {candidate.original_speaker_id: candidate for candidate in unresolved}
        for result in decision.roles:
            candidate = candidates_by_id[result.original_speaker_id]
            evidence_segment_ids = _assignment_evidence_segment_ids(candidate, result)
            assignments[result.original_speaker_id] = RoleAssignment(
                original_speaker_id=result.original_speaker_id,
                role=result.role,
                confidence=result.confidence,
                evidence_segment_ids=evidence_segment_ids,
                source=RoleResolutionSource.AGENT,
                resolution_evidence=result.evidence,
            )

    unresolved_candidates = tuple(
        candidate for candidate in candidates if candidate.original_speaker_id not in assignments
    )
    if unresolved_candidates:
        raise RoleResolutionRequiredError(unresolved_candidates)
    if len(assignments) == 2 and {assignment.role for assignment in assignments.values()} != {
        SpeakerRole.OPERATOR,
        SpeakerRole.CLIENT,
    }:
        raise RoleResolutionRequiredError(candidates)

    ordered = tuple(assignments[candidate.original_speaker_id] for candidate in candidates)
    return RoleResolution(
        assignments=ordered,
        needs_review=any(item.confidence < review_confidence_threshold for item in ordered),
        agent_provenance=prompt_provenance,
    )


def _assignment_evidence_segment_ids(
    candidate: RoleResolutionCandidate,
    result: ResolvedRole,
) -> tuple[UUID, ...]:
    if not result.evidence_segment_ids:
        raise ValueError("role agent должен вернуть точные evidence segment IDs")
    if not set(result.evidence_segment_ids).issubset(candidate.evidence_segment_ids):
        raise ValueError("role agent вернул evidence segment ID другого speaker")
    return result.evidence_segment_ids


def _candidates(segments: tuple[DiarizedSegment, ...]) -> tuple[RoleResolutionCandidate, ...]:
    grouped: dict[str, list[DiarizedSegment]] = {}
    for segment in segments:
        grouped.setdefault(segment.original_speaker_id, []).append(segment)

    candidates: list[RoleResolutionCandidate] = []
    for speaker_id, speaker_segments in grouped.items():
        confidences = tuple(item.speaker_confidence for item in speaker_segments if item.speaker_confidence is not None)
        candidates.append(
            RoleResolutionCandidate(
                original_speaker_id=speaker_id,
                evidence_segment_ids=tuple(item.id for item in speaker_segments),
                evidence_segments=tuple(
                    RoleSegmentEvidence(segment_id=item.id, text=item.text) for item in speaker_segments
                ),
                speaker_confidence=max(confidences) if confidences else None,
            )
        )
    return tuple(candidates)


def _validate_resolver_output(
    candidates: Iterable[RoleResolutionCandidate],
    resolved: tuple[ResolvedRole, ...],
) -> None:
    candidate_ids = {candidate.original_speaker_id for candidate in candidates}
    resolved_ids = tuple(item.original_speaker_id for item in resolved)
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("role agent вернул повторяющиеся original speaker IDs")
    if not set(resolved_ids).issubset(candidate_ids):
        raise ValueError("role agent вернул неизвестный original speaker ID")
