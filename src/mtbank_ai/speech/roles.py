"""Явное content-based разрешение ролей без эвристики порядка спикеров."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from mtbank_ai.domain.transcript import (
    RoleAssignment,
    RolePolicyProvenance,
    RoleResolution,
    RoleResolutionSource,
    SpeakerRole,
)
from mtbank_ai.policies.loader import LoadedPolicyPack, RoleSignal, RolesPolicy
from mtbank_ai.speech.contracts import (
    DiarizedSegment,
    ResolvedRole,
    RoleResolutionCandidate,
    RoleResolutionDecision,
    RoleSegmentEvidence,
    SpeakerRoleMapping,
)


class RoleResolverPort(Protocol):
    """Ограниченный injected port для доменного resolver-а ролей."""

    def resolve(self, candidates: tuple[RoleResolutionCandidate, ...]) -> RoleResolutionDecision: ...


class RoleResolutionRequiredError(Exception):
    """Ни metadata, ни resolver не покрыли все diarization labels до snapshot."""

    def __init__(self, candidates: tuple[RoleResolutionCandidate, ...]) -> None:
        self.candidates = candidates
        super().__init__("role_resolution_required")


@dataclass(frozen=True, slots=True)
class _RoleScore:
    value: float
    evidence_segment_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class _RoleScores:
    operator: _RoleScore
    client: _RoleScore

    def for_role(self, role: SpeakerRole) -> _RoleScore:
        return self.operator if role is SpeakerRole.OPERATOR else self.client


class PolicyRoleResolver:
    """Детерминированный resolver по immutable verified roles policy pack."""

    def __init__(self, policy_pack: LoadedPolicyPack[RolesPolicy]) -> None:
        if policy_pack.name != "roles":
            raise ValueError("role resolver требует roles policy pack")
        self._policy_pack = policy_pack
        metadata = policy_pack.policy.metadata
        self._policy_provenance = RolePolicyProvenance(
            policy_id=metadata.policy_id,
            version=metadata.version,
            owner=metadata.owner,
            effective_date=metadata.effective_date,
            sha256=policy_pack.sha256,
        )

    @property
    def policy_provenance(self) -> RolePolicyProvenance:
        return self._policy_provenance

    @property
    def supported_automatic_total_speakers(self) -> int:
        return self._policy_pack.policy.thresholds.supported_automatic_total_speakers

    @property
    def review_confidence_threshold(self) -> float:
        return self._policy_pack.policy.thresholds.review_confidence_threshold

    def resolve(self, candidates: tuple[RoleResolutionCandidate, ...]) -> RoleResolutionDecision:
        if len(candidates) == 1:
            roles = self._resolve_one(candidates[0])
        elif len(candidates) == self.supported_automatic_total_speakers:
            roles = self._resolve_pair(candidates)
        else:
            roles = ()
        return RoleResolutionDecision(
            roles=roles,
            policy_provenance=self._policy_provenance if roles else None,
        )

    def _resolve_one(self, candidate: RoleResolutionCandidate) -> tuple[ResolvedRole, ...]:
        scores = self._scores(candidate)
        role = SpeakerRole.OPERATOR if scores.operator.value > scores.client.value else SpeakerRole.CLIENT
        score = scores.for_role(role)
        margin = abs(scores.operator.value - scores.client.value)
        minimum_margin = self._policy_pack.policy.thresholds.minimum_full_assignment_margin
        if score.value < self._minimum_score(role) or margin < minimum_margin:
            return ()
        return (self._resolved(candidate, role, score, margin),)

    def _resolve_pair(self, candidates: tuple[RoleResolutionCandidate, ...]) -> tuple[ResolvedRole, ...]:
        first, second = candidates
        first_scores = self._scores(first)
        second_scores = self._scores(second)
        operator_first_total = first_scores.operator.value + second_scores.client.value
        client_first_total = first_scores.client.value + second_scores.operator.value
        operator_first_valid = (
            first_scores.operator.value >= self._minimum_score(SpeakerRole.OPERATOR)
            and second_scores.client.value >= self._minimum_score(SpeakerRole.CLIENT)
        )
        client_first_valid = (
            first_scores.client.value >= self._minimum_score(SpeakerRole.CLIENT)
            and second_scores.operator.value >= self._minimum_score(SpeakerRole.OPERATOR)
        )
        if not operator_first_valid and not client_first_valid:
            return ()
        margin = abs(operator_first_total - client_first_total)
        if margin < self._policy_pack.policy.thresholds.minimum_full_assignment_margin:
            return ()
        if operator_first_valid and (not client_first_valid or operator_first_total > client_first_total):
            return (
                self._resolved(first, SpeakerRole.OPERATOR, first_scores.operator, margin),
                self._resolved(second, SpeakerRole.CLIENT, second_scores.client, margin),
            )
        if client_first_valid and (not operator_first_valid or client_first_total > operator_first_total):
            return (
                self._resolved(first, SpeakerRole.CLIENT, first_scores.client, margin),
                self._resolved(second, SpeakerRole.OPERATOR, second_scores.operator, margin),
            )
        return ()

    def _scores(self, candidate: RoleResolutionCandidate) -> _RoleScores:
        signals = self._policy_pack.policy.signals
        return _RoleScores(
            operator=_signals_score(candidate.evidence_segments, signals.operator),
            client=_signals_score(candidate.evidence_segments, signals.client),
        )

    def _minimum_score(self, role: SpeakerRole) -> float:
        thresholds = self._policy_pack.policy.thresholds
        return thresholds.minimum_operator_score if role is SpeakerRole.OPERATOR else thresholds.minimum_client_score

    def _resolved(
        self,
        candidate: RoleResolutionCandidate,
        role: SpeakerRole,
        score: _RoleScore,
        margin: float,
    ) -> ResolvedRole:
        formula = self._policy_pack.policy.confidence
        score_component = min(score.value / formula.score_scale, 1.0)
        margin_component = min(margin / formula.margin_scale, 1.0)
        confidence = formula.minimum + (formula.maximum - formula.minimum) * (
            formula.score_weight * score_component + formula.margin_weight * margin_component
        )
        confidence = min(max(confidence, formula.minimum), formula.maximum)
        return ResolvedRole(
            original_speaker_id=candidate.original_speaker_id,
            role=role,
            confidence=confidence,
            evidence=f"{self._policy_pack.version}/{self._policy_pack.sha256}",
            evidence_segment_ids=score.evidence_segment_ids,
        )


def resolve_roles(
    segments: tuple[DiarizedSegment, ...],
    *,
    metadata_mappings: tuple[SpeakerRoleMapping, ...] = (),
    resolver: RoleResolverPort | None = None,
    review_confidence_threshold: float = 0.75,
) -> RoleResolution:
    """Resolve all labels before creating TranscriptSnapshot or raise a typed review outcome."""

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

    policy_provenance: RolePolicyProvenance | None = None
    if unresolved and resolver is not None and _supports_candidate_count(resolver, len(candidates)):
        decision = _as_decision(resolver.resolve(tuple(unresolved)))
        _validate_resolver_output(unresolved, decision.roles)
        candidates_by_id = {candidate.original_speaker_id: candidate for candidate in unresolved}
        policy_provenance = decision.policy_provenance
        source = RoleResolutionSource.POLICY if policy_provenance is not None else RoleResolutionSource.RESOLVER
        for result in decision.roles:
            candidate = candidates_by_id[result.original_speaker_id]
            evidence_segment_ids = _assignment_evidence_segment_ids(candidate, result, policy_provenance)
            assignments[result.original_speaker_id] = RoleAssignment(
                original_speaker_id=result.original_speaker_id,
                role=result.role,
                confidence=result.confidence,
                evidence_segment_ids=evidence_segment_ids,
                source=source,
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
    threshold = review_confidence_threshold
    if policy_provenance is not None and isinstance(resolver, PolicyRoleResolver):
        threshold = resolver.review_confidence_threshold
    return RoleResolution(
        assignments=ordered,
        needs_review=any(item.confidence < threshold for item in ordered),
        policy_provenance=policy_provenance,
    )


def _as_decision(result: object) -> RoleResolutionDecision:
    if isinstance(result, RoleResolutionDecision):
        return result
    if isinstance(result, tuple) and all(isinstance(item, ResolvedRole) for item in result):
        return RoleResolutionDecision(roles=result)
    raise ValueError("role resolver должен вернуть typed role resolution decision")


def _assignment_evidence_segment_ids(
    candidate: RoleResolutionCandidate,
    result: ResolvedRole,
    policy_provenance: RolePolicyProvenance | None,
) -> tuple[UUID, ...]:
    if not result.evidence_segment_ids:
        if policy_provenance is not None:
            raise ValueError("policy role resolver должен вернуть точные evidence segment IDs")
        return candidate.evidence_segment_ids
    if not set(result.evidence_segment_ids).issubset(candidate.evidence_segment_ids):
        raise ValueError("role resolver вернул evidence segment ID другого speaker")
    return result.evidence_segment_ids


def _normalize_evidence(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join("".join(character if character.isalnum() else " " for character in normalized).split())


def _signals_score(
    evidence_segments: tuple[RoleSegmentEvidence, ...],
    signals: tuple[RoleSignal, ...],
) -> _RoleScore:
    normalized_segments = tuple((item.segment_id, _normalize_evidence(item.text)) for item in evidence_segments)
    score = 0.0
    evidence_segment_ids: list[UUID] = []
    for signal in signals:
        phrases = tuple(_normalize_evidence(phrase) for phrase in signal.phrases)
        matched_ids = tuple(
            segment_id
            for segment_id, text in normalized_segments
            if any(phrase and f" {phrase} " in f" {text} " for phrase in phrases)
        )
        if not matched_ids:
            continue
        score += signal.weight
        for segment_id in matched_ids:
            if segment_id not in evidence_segment_ids:
                evidence_segment_ids.append(segment_id)
    return _RoleScore(value=score, evidence_segment_ids=tuple(evidence_segment_ids))


def _supports_candidate_count(resolver: RoleResolverPort, candidate_count: int) -> bool:
    if not isinstance(resolver, PolicyRoleResolver):
        return True
    return candidate_count == resolver.supported_automatic_total_speakers


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
        raise ValueError("role resolver вернул повторяющиеся original speaker IDs")
    if not set(resolved_ids).issubset(candidate_ids):
        raise ValueError("role resolver вернул неизвестный original speaker ID")
