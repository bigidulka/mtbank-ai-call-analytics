#!/usr/bin/env python3
"""Create privacy-safe aggregate evidence from a local external speech benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GROUPS = {
    "axon-real-calls": "axon-",
    "fleurs-read-speech": "fleurs-",
    "golos-farfield": "golos-",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(
    files: list[dict[str, object]],
    *,
    configured_start_padding: float,
    configured_end_padding: float,
) -> dict[str, object]:
    substitutions = deletions = insertions = reference_words = 0
    audio_seconds = asr_ms = luna_ms = vad_ms = 0.0
    fixed_padding_compatible = adaptive_padding_used = failed_alignments = 0
    for item in files:
        media = item["media"]
        asr = item["asr"]
        luna = item["luna"]
        vad = item["vad"]
        assert isinstance(media, dict) and isinstance(asr, dict) and isinstance(luna, dict) and isinstance(vad, dict)
        counts = asr["wer"]
        assert isinstance(counts, dict)
        substitutions += int(counts["substitutions"])
        deletions += int(counts["deletions"])
        insertions += int(counts["insertions"])
        reference_words += int(counts["reference_words"])
        audio_seconds += float(media["duration_seconds"])
        asr_ms += float(asr["latency_ms"])
        luna_ms += float(luna["latency_ms"])
        vad_ms += float(vad["latency_ms"])
        status = vad.get("status")
        if status != "completed":
            failed_alignments += 1
            continue
        applied_start = float(vad["applied_start_padding_seconds"])
        applied_end = float(vad["applied_end_padding_seconds"])
        reduced = bool(vad.get("padding_reduced_to_avoid_overlap"))
        if not reduced and applied_start == configured_start_padding and applied_end == configured_end_padding:
            fixed_padding_compatible += 1
        elif reduced and (applied_start, applied_end) in {(0.1, 0.2), (0.0, 0.0)}:
            adaptive_padding_used += 1
        else:
            raise ValueError("alignment padding/status classification is inconsistent")
    errors = substitutions + deletions + insertions
    return {
        "files": len(files),
        "audio_seconds": audio_seconds,
        "wer": errors / reference_words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": reference_words,
        "asr_latency_ms": asr_ms,
        "luna_latency_ms": luna_ms,
        "vad_alignment_latency_ms": vad_ms,
        "sequential_component_latency_ms": asr_ms + luna_ms + vad_ms,
        "fixed_padding_compatible_files": fixed_padding_compatible,
        "adaptive_padding_used_files": adaptive_padding_used,
        "alignment_failed_files": failed_alignments,
    }


def sanitize(input_path: Path) -> dict[str, object]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    files = raw.get("files")
    if raw.get("status") != "completed" or not isinstance(files, list) or not files:
        raise ValueError("benchmark result is incomplete")
    typed_files = [item for item in files if isinstance(item, dict)]
    if len(typed_files) != len(files):
        raise ValueError("benchmark result files are malformed")
    configuration = raw.get("configuration")
    provenance = raw.get("provenance")
    if not isinstance(configuration, dict) or not isinstance(provenance, dict):
        raise ValueError("benchmark provenance/configuration is missing")
    configured_start = float(configuration["speech_start_padding_seconds"])
    configured_end = float(configuration["speech_end_padding_seconds"])
    grouped = {
        name: _aggregate(
            [item for item in typed_files if str(item.get("id", "")).startswith(prefix)],
            configured_start_padding=configured_start,
            configured_end_padding=configured_end,
        )
        for name, prefix in GROUPS.items()
    }
    if any(group["files"] == 0 for group in grouped.values()):
        raise ValueError("benchmark source group is empty")
    overall = _aggregate(
        typed_files,
        configured_start_padding=configured_start,
        configured_end_padding=configured_end,
    )
    source_revisions = sorted(
        {str(source.get("provenance")) for item in typed_files if isinstance((source := item.get("source")), dict)}
    )
    return {
        "schema_version": 1,
        "kind": "privacy-safe-external-speech-benchmark-summary",
        "status": "completed",
        "scope": (
            "aggregate metrics only; raw external audio, transcripts, hypotheses, names, and source documents excluded"
        ),
        "local_run": {
            "input_sha256": _sha256(input_path),
            "source_manifest_sha256": provenance["manifest_sha256"],
            "evaluator_sha256": provenance["evaluator_sha256"],
            "prompt_hash": provenance["prompt_hash"],
            "requested_asr_model": provenance["requested_asr_model"],
            "actual_asr_model": provenance["actual_asr_model"],
            "luna_model": provenance["luna_model"],
            "configuration": configuration,
            "source_revisions": source_revisions,
        },
        "groups": grouped,
        "overall": {
            **overall,
            "wall_latency_ms": raw["aggregate"]["wall_latency_ms"],
            "real_time_factor": raw["aggregate"]["real_time_factor"],
        },
        "claims": {
            "role_accuracy_or_der": "not measurable: sources lack trusted operator/client turn timestamps",
            "fixed_padding": (
                "0.2s start / 0.9s end fails on both long natural calls; adaptive 0.1/0.2 diagnostic completed"
            ),
            "privacy": (
                "public artifact intentionally excludes raw call audio and content; public download/license did not "
                "establish voice/privacy redistribution consent"
            ),
            "latency": (
                "observed wall/component values from one sequential run; not reproducible offline and not an SLA "
                "attestation"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = sanitize(arguments.input)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
