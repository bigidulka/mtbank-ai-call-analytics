from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from mtbank_ai.domain.transcript import RoleResolutionSource, SpeakerRole
from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.speech.contracts import (
    DiarizedSegment,
    RoleResolutionCandidate,
    RoleResolutionDecision,
    RoleSegmentEvidence,
    SpeakerRoleMapping,
)
from mtbank_ai.speech.roles import PolicyRoleResolver, RoleResolutionRequiredError, resolve_roles
from scripts.evaluate_speech import Segment, speaker_attributed_wer, time_weighted_role_accuracy

ROOT = Path(__file__).parents[3]


def _resolver() -> PolicyRoleResolver:
    return PolicyRoleResolver(PolicyRegistry().roles)


def _segments(values: tuple[tuple[str, str], ...]) -> tuple[DiarizedSegment, ...]:
    return tuple(
        DiarizedSegment(
            id=uuid5(NAMESPACE_URL, f"policy-role-test/{speaker_id}/{index}"),
            original_speaker_id=speaker_id,
            start=float(index),
            end=float(index + 1),
            text=text,
        )
        for index, (speaker_id, text) in enumerate(values)
    )


def _reference_segments(path: Path, *, client_first: bool = False) -> tuple[DiarizedSegment, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = payload["segments"]
    assert isinstance(raw_segments, list)
    segments = tuple(
        DiarizedSegment(
            id=uuid5(NAMESPACE_URL, item["id"]),
            original_speaker_id="speaker-operator" if item["speaker"] == "Оператор" else "speaker-client",
            start=item["start"],
            end=item["end"],
            text=item["text"],
        )
        for item in raw_segments
    )
    if not client_first:
        return segments
    return tuple(segment for segment in segments if segment.original_speaker_id == "speaker-client") + tuple(
        segment for segment in segments if segment.original_speaker_id == "speaker-operator"
    )


def _assigned_roles(segments: tuple[DiarizedSegment, ...]) -> dict[str, SpeakerRole]:
    resolution = resolve_roles(segments, resolver=_resolver())
    return {assignment.original_speaker_id: assignment.role for assignment in resolution.assignments}


def test_policy_resolver_is_content_based_client_first_and_records_exact_provenance() -> None:
    policy = PolicyRegistry().roles
    resolution = resolve_roles(
        _segments(
            (
                ("speaker-client", "Хочу узнать статус перевода, спасибо."),
                ("speaker-operator", "МТБанк, оператор Анна. Чем могу помочь?"),
            )
        ),
        resolver=PolicyRoleResolver(policy),
    )

    assignments = {assignment.original_speaker_id: assignment for assignment in resolution.assignments}
    assert assignments["speaker-client"].role is SpeakerRole.CLIENT
    assert assignments["speaker-operator"].role is SpeakerRole.OPERATOR
    assert {assignment.source for assignment in resolution.assignments} == {RoleResolutionSource.POLICY}
    assert resolution.policy_provenance is not None
    assert resolution.policy_provenance.policy_id == policy.policy.metadata.policy_id
    assert resolution.policy_provenance.version == policy.version
    assert resolution.policy_provenance.owner == policy.owner
    assert resolution.policy_provenance.effective_date == policy.effective_date
    assert resolution.policy_provenance.sha256 == policy.sha256
    assert not resolution.needs_review


def test_policy_role_port_returns_typed_decision_with_exact_provenance() -> None:
    operator_id = uuid5(NAMESPACE_URL, "policy-port/operator")
    client_id = uuid5(NAMESPACE_URL, "policy-port/client")
    decision = _resolver().resolve(
        (
            RoleResolutionCandidate(
                original_speaker_id="speaker-operator",
                evidence_segment_ids=(operator_id,),
                evidence_segments=(
                    RoleSegmentEvidence(
                        segment_id=operator_id,
                        text="МТБанк, оператор Анна. Чем могу помочь?",
                    ),
                ),
            ),
            RoleResolutionCandidate(
                original_speaker_id="speaker-client",
                evidence_segment_ids=(client_id,),
                evidence_segments=(
                    RoleSegmentEvidence(
                        segment_id=client_id,
                        text="Хочу уточнить перевод, спасибо.",
                    ),
                ),
            ),
        )
    )

    assert isinstance(decision, RoleResolutionDecision)
    assert {role.role for role in decision.roles} == {SpeakerRole.OPERATOR, SpeakerRole.CLIENT}
    assert decision.policy_provenance is not None
    assert decision.policy_provenance.sha256 == PolicyRegistry().roles.sha256


@pytest.mark.parametrize(
    "segments",
    (
        _segments(
            (
                ("speaker-a", "МТБанк, хочу получить помощь."),
                ("speaker-b", "МТБанк, хочу получить помощь."),
            )
        ),
        _segments(
            (
                ("speaker-a", "Нейтральная реплика без signal."),
                ("speaker-b", "Ещё одна нейтральная реплика."),
            )
        ),
        _segments(
            (
                ("speaker-a", "МТБанк, оператор."),
                ("speaker-b", "Хочу получить помощь."),
                ("speaker-c", "Спасибо."),
            )
        ),
    ),
)
def test_policy_resolver_fails_closed_for_tie_insufficient_evidence_and_unsupported_count(
    segments: tuple[DiarizedSegment, ...],
) -> None:
    with pytest.raises(RoleResolutionRequiredError) as error:
        resolve_roles(segments, resolver=_resolver())

    assert {candidate.original_speaker_id for candidate in error.value.candidates} == {
        segment.original_speaker_id for segment in segments
    }


def test_metadata_mapping_keeps_priority_while_policy_resolver_covers_remaining_speaker() -> None:
    resolution = resolve_roles(
        _segments(
            (
                ("speaker-operator", "МТБанк, оператор Анна. Чем могу помочь?"),
                ("speaker-client", "Хочу уточнить перевод, спасибо."),
            )
        ),
        metadata_mappings=(
            SpeakerRoleMapping(
                original_speaker_id="speaker-operator",
                role=SpeakerRole.OPERATOR,
                evidence="trusted-call-metadata",
            ),
        ),
        resolver=_resolver(),
    )

    assignments = {assignment.original_speaker_id: assignment for assignment in resolution.assignments}
    assert assignments["speaker-operator"].source is RoleResolutionSource.METADATA
    assert assignments["speaker-operator"].resolution_evidence == "trusted-call-metadata"
    assert assignments["speaker-client"].source is RoleResolutionSource.POLICY
    assert assignments["speaker-client"].role is SpeakerRole.CLIENT
    assert resolution.policy_provenance is not None


@pytest.mark.parametrize("reference_path", tuple(sorted((ROOT / "test_data" / "references").glob("*.json"))))
def test_policy_resolver_assigns_all_reference_fixture_roles_order_independently(reference_path: Path) -> None:
    expected = {
        "speaker-operator": SpeakerRole.OPERATOR,
        "speaker-client": SpeakerRole.CLIENT,
    }

    assert _assigned_roles(_reference_segments(reference_path)) == expected
    assert _assigned_roles(_reference_segments(reference_path, client_first=True)) == expected


@pytest.mark.parametrize("reference_path", tuple(sorted((ROOT / "test_data" / "references").glob("*.json"))))
def test_policy_resolver_preserves_role_evaluation_metrics_for_all_reference_fixtures(reference_path: Path) -> None:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    raw_segments = payload["segments"]
    assert isinstance(raw_segments, list)
    roles = _assigned_roles(_reference_segments(reference_path, client_first=True))
    reference = tuple(
        Segment(item["id"], item["start"], item["end"], item["speaker"], item["text"])
        for item in raw_segments
    )
    hypothesis = tuple(
        Segment(
            item["id"],
            item["start"],
            item["end"],
            roles["speaker-operator" if item["speaker"] == "Оператор" else "speaker-client"].value,
            item["text"],
        )
        for item in raw_segments
    )

    assert time_weighted_role_accuracy(reference, hypothesis) == 1.0
    assert speaker_attributed_wer(reference, hypothesis).errors == 0


def test_policy_resolver_records_only_segments_matching_selected_role_signals() -> None:
    segments = _segments(
        (
            ("speaker-client", "Нейтральная реплика."),
            ("speaker-operator", "МТБанк, оператор Анна. Чем могу помочь?"),
            ("speaker-client", "Хочу уточнить перевод, спасибо."),
            ("speaker-operator", "Нейтральная служебная реплика."),
        )
    )

    resolution = resolve_roles(segments, resolver=_resolver())
    assignments = {assignment.original_speaker_id: assignment for assignment in resolution.assignments}

    assert assignments["speaker-operator"].evidence_segment_ids == (segments[1].id,)
    assert assignments["speaker-client"].evidence_segment_ids == (segments[2].id,)


def test_policy_resolver_returns_same_mapping_after_candidate_order_swap() -> None:
    original = _segments(
        (
            ("speaker-operator", "МТБанк, оператор Анна. Чем могу помочь?"),
            ("speaker-client", "Хочу уточнить перевод, спасибо."),
        )
    )
    swapped = (original[1], original[0])

    assert _assigned_roles(original) == _assigned_roles(swapped)


def test_duplicate_two_speaker_metadata_mapping_fails_closed() -> None:
    segments = _segments(
        (
            ("speaker-operator", "МТБанк, оператор Анна. Чем могу помочь?"),
            ("speaker-client", "Хочу уточнить перевод, спасибо."),
        )
    )

    with pytest.raises(RoleResolutionRequiredError):
        resolve_roles(
            segments,
            metadata_mappings=(
                SpeakerRoleMapping(
                    original_speaker_id="speaker-operator",
                    role=SpeakerRole.OPERATOR,
                    evidence="trusted-metadata/operator",
                ),
                SpeakerRoleMapping(
                    original_speaker_id="speaker-client",
                    role=SpeakerRole.OPERATOR,
                    evidence="trusted-metadata/client",
                ),
            ),
            resolver=_resolver(),
        )
