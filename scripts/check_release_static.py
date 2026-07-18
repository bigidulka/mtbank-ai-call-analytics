#!/usr/bin/env python3
"""Детерминированные offline проверки секретов и dependency lockfiles."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
_SECRET_ASSIGNMENT = re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]([^'\"]+)['\"]")
_ALLOWED_TEST_MARKERS = ("tests/", ".example", "test_data/")


def check_secrets() -> list[str]:
    completed = subprocess.run(
        ("git", "-C", str(ROOT), "ls-files", "-z"),
        check=True,
        capture_output=True,
    )
    failures: list[str] = []
    for encoded_path in completed.stdout.split(b"\0"):
        if not encoded_path:
            continue
        relative = encoded_path.decode("utf-8")
        if relative.startswith(_ALLOWED_TEST_MARKERS):
            continue
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = _SECRET_ASSIGNMENT.search(line)
            if match and not match.group(2).startswith("${"):
                failures.append(f"{relative}:{line_number}: возможный literal secret")
    return failures


def check_locks() -> list[str]:
    required = (ROOT / "uv.lock", ROOT / "services" / "speech" / "uv.lock")
    return [f"не найден lockfile: {path.relative_to(ROOT)}" for path in required if not path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("secrets", "locks"))
    arguments = parser.parse_args()
    failures = check_secrets() if arguments.check == "secrets" else check_locks()
    if failures:
        print("\n".join(failures))
        return 1
    print(f"{arguments.check}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
