from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.policies.loader import RolesPolicy

ROOT = Path(__file__).parents[2]


def _policy_payload() -> dict[str, object]:
    return json.loads((ROOT / "src" / "mtbank_ai" / "policies" / "roles" / "v1.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["signals"]["operator"][0]["phrases"].append("МТБАНК!!!"),
        lambda payload: payload["signals"]["client"][0]["phrases"].append("мтбанк"),
        lambda payload: payload["thresholds"].__setitem__("review_confidence_threshold", 0.0),
    ),
)
def test_roles_policy_rejects_ambiguous_normalized_phrases_and_unsound_thresholds(mutation) -> None:
    payload = copy.deepcopy(_policy_payload())
    mutation(payload)

    with pytest.raises(ValidationError):
        RolesPolicy.model_validate(payload, strict=True)


def test_verified_roles_policy_remains_speech_only_pack() -> None:
    registry = PolicyRegistry()

    assert registry.roles.policy.metadata.policy_id == "roles"
    assert tuple(pack.name for pack in registry.load_all()) == ("taxonomy", "quality", "compliance")
