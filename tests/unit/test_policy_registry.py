from __future__ import annotations

from mtbank_ai.policies import PolicyRegistry


def test_registry_load_all_excludes_speech_only_roles_policy() -> None:
    packs = PolicyRegistry().load_all()

    assert tuple(pack.name for pack in packs) == ("taxonomy", "quality", "compliance")
