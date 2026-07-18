#!/usr/bin/env python3
"""Печатает release-gate manifest и не пропускает блокированную поставку."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mtbank_ai.release.gates import ReleaseGateContext, evaluate_release_gate

ROOT = Path(__file__).parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-blocked", action="store_true", help="только сформировать отчёт, не разрешая release")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    manifest = evaluate_release_gate(ReleaseGateContext.from_process(ROOT))
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0 if arguments.allow_blocked or manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
