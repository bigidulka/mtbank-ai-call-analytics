#!/usr/bin/env python3
"""Boundary timing diagnostics for non-overlapping two-party turn hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from mtbank_ai.speech.dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("boundary metric requires values")
    position = (len(ordered) - 1) * percentile
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    payload = json.loads(arguments.hypotheses.read_text())
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ValueError("hypothesis evidence is not completed")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("hypothesis evidence requires files list")
    entries = tuple(
        entry
        for entry in validate_manifest(arguments.manifest, require_release_corpus=True)
        if entry.kind == "speech_reference"
    )
    expected_ids = tuple(entry.identifier for entry in entries)
    actual_ids = tuple(str(item.get("id")) for item in raw_files if isinstance(item, dict))
    if actual_ids != expected_ids or len(actual_ids) != len(raw_files) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("hypothesis evidence must exactly cover manifest entries in order")
    hypotheses = {str(item["id"]): item for item in raw_files}
    errors: list[float] = []
    signed: list[float] = []
    files = []
    for entry in entries:
        reference = json.loads((entry.root / str(entry.raw["reference_path"])).read_text())["segments"]
        hypothesis_file = hypotheses[entry.identifier]
        hypothesis = hypothesis_file["segments"]
        duration = float(entry.raw["duration_seconds"])
        if len(hypothesis) != len(reference):
            raise ValueError(f"{entry.identifier}: segment counts differ")
        previous_end = -1.0
        for segment in hypothesis:
            start, end = float(segment["start"]), float(segment["end"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end or end > duration:
                raise ValueError(f"{entry.identifier}: invalid hypothesis interval")
            if start < previous_end:
                raise ValueError(f"{entry.identifier}: hypothesis intervals overlap or reorder")
            previous_end = end
        reference_boundaries = [
            (float(left["end"]) + float(right["start"])) / 2 for left, right in zip(reference, reference[1:])
        ]
        hypothesis_boundaries = [
            (float(left["end"]) + float(right["start"])) / 2 for left, right in zip(hypothesis, hypothesis[1:])
        ]
        if len(reference_boundaries) != len(hypothesis_boundaries):
            raise ValueError(f"{entry.identifier}: boundary counts differ")
        current_signed = [
            actual - expected for expected, actual in zip(reference_boundaries, hypothesis_boundaries, strict=True)
        ]
        current = [abs(value) for value in current_signed]
        errors.extend(current)
        signed.extend(current_signed)
        files.append(
            {
                "id": entry.identifier,
                "boundaries": len(current),
                "median_absolute_error_seconds": _percentile(current, 0.5),
                "p95_absolute_error_seconds": _percentile(current, 0.95),
                "mean_signed_error_seconds": sum(current_signed) / len(current_signed),
            }
        )
    return {
        "schema_version": 1,
        "kind": "turn-boundary-evaluation",
        "status": "completed",
        "provenance": {
            "manifest_path": str(arguments.manifest),
            "manifest_sha256": _sha256(arguments.manifest),
            "hypotheses_path": str(arguments.hypotheses),
            "hypotheses_sha256": _sha256(arguments.hypotheses),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
        },
        "files": files,
        "micro": {
            "boundaries": len(errors),
            "median_absolute_error_seconds": _percentile(errors, 0.5),
            "p95_absolute_error_seconds": _percentile(errors, 0.95),
            "mean_absolute_error_seconds": sum(errors) / len(errors),
            "mean_signed_error_seconds": sum(signed) / len(signed),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "turn-boundary-evaluation",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
