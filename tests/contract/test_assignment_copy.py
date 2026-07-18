from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).parents[2]
ASSIGNMENT_SHA256 = "8a63f1e2078dd97bfa69d4bf72cd968d0196830584e9faefed37135afbe334f6"


def test_original_assignment_is_preserved_byte_for_byte() -> None:
    assignment = (ROOT / "docs" / "assignment.md").read_bytes()

    assert len(assignment) == 15_970
    assert hashlib.sha256(assignment).hexdigest() == ASSIGNMENT_SHA256
