#!/usr/bin/env python3
"""Build one provenance-linked comparison of candidate and canonical speech evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mtbank_ai.speech.dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(arguments: argparse.Namespace) -> dict[str, object]:
    candidate = json.loads(arguments.candidate.read_text(encoding="utf-8"))
    canonical = json.loads(arguments.canonical.read_text(encoding="utf-8"))
    if candidate.get("status") != "completed" or canonical.get("status") != "completed":
        raise ValueError("candidate and canonical evidence must be completed")
    entries = tuple(
        item
        for item in validate_manifest(arguments.manifest, require_release_corpus=True)
        if item.kind == "speech_reference"
    )
    expected_ids = tuple(item.identifier for item in entries)
    manifest = arguments.manifest.read_bytes()
    candidate_ids = tuple(file["id"] for file in candidate["files"])
    canonical_ids = tuple(file["id"] for file in canonical["files"])
    if (
        candidate_ids != expected_ids
        or canonical_ids != expected_ids
        or len(set(candidate_ids)) != len(candidate_ids)
        or len(set(canonical_ids)) != len(canonical_ids)
    ):
        raise ValueError("candidate and canonical evidence must exactly cover manifest entries in order")
    expected_manifest_hash = hashlib.sha256(manifest).hexdigest()
    if canonical.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("canonical manifest hash differs from current corpus")
    candidate_provenance = candidate.get("provenance")
    if (
        not isinstance(candidate_provenance, dict)
        or candidate_provenance.get("manifest_sha256") != expected_manifest_hash
    ):
        raise ValueError("candidate manifest hash differs from current corpus")
    expected_hashes = tuple(str(item.raw["sha256"]) for item in entries)
    candidate_hashes = tuple(file.get("audio_sha256") for file in candidate["files"])
    canonical_hashes = tuple(file.get("audio_sha256") for file in canonical["files"])
    if candidate_hashes != expected_hashes or canonical_hashes != expected_hashes:
        raise ValueError("candidate or canonical audio hashes differ from manifest")
    return {
        "schema_version": 1,
        "kind": "speech-profile-comparison",
        "status": "completed",
        "scope": "authored synthetic/no-PII corpus only",
        "manifest_sha256": expected_manifest_hash,
        "ordered_file_ids": candidate_ids,
        "evidence": {
            "candidate": {
                "path": str(arguments.candidate),
                "sha256": _sha256(arguments.candidate),
                "provenance": candidate_provenance,
            },
            "canonical": {"path": str(arguments.canonical), "sha256": _sha256(arguments.canonical)},
        },
        "candidate": candidate["micro"],
        "canonical": canonical["micro"],
        "claim_boundary": (
            "Comparable corpus hashes and shared metric functions; candidate exploits authored "
            "fixed-pause TTS structure "
            "and has no independent natural-call evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--canonical", type=Path, default=ROOT / "release-evidence/final-115/canonical-speech-evaluation.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = compare(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "speech-profile-comparison",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
