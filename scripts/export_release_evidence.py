#!/usr/bin/env python3
"""Экспортирует privacy-safe release evidence без content-bearing payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mtbank_ai.release.evidence import export_evidence


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} должен быть JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--code-sha", required=True)
    arguments = parser.parse_args()

    source: Any = json.loads(arguments.input.read_text(encoding="utf-8"))
    source = _mapping(source, "input")
    evidence = export_evidence(
        kind=arguments.kind,
        code_sha=arguments.code_sha,
        provenance=_mapping(source.get("provenance", {}), "provenance"),
        metrics=_mapping(source.get("metrics", {}), "metrics"),
        observations=_mapping(source.get("observations", {}), "observations"),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
