#!/usr/bin/env python3
"""Verify privacy-safe external speech benchmark aggregate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_EVALUATOR_SHA256 = "61083df88759fad8fcdd4805c003965d579769216e66fe81e7a078d72a432369"
RUN_SOURCE_MANIFEST_SHA256 = "70fdaec63ec5db4ac926aabf76cbb59298a41a3bb377ef1b9b46704fb164b7f5"
EXPECTED_GROUPS = {
    "axon-real-calls": {"files": 2, "audio_seconds": 894.312, "reference_words": 1669},
    "fleurs-read-speech": {"files": 5, "audio_seconds": 67.62, "reference_words": 108},
    "golos-farfield": {"files": 5, "audio_seconds": 13.564939, "reference_words": 24},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_number(value: object, name: str, *, nonnegative: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _validate_group(name: str, group: object) -> None:
    if not isinstance(group, dict):
        raise ValueError(f"{name} group is invalid")
    expected = EXPECTED_GROUPS[name]
    if group.get("files") != expected["files"] or group.get("reference_words") != expected["reference_words"]:
        raise ValueError(f"{name} coverage changed")
    if not math.isclose(
        _require_number(group.get("audio_seconds"), f"{name}.audio_seconds"),
        float(expected["audio_seconds"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name} duration changed")
    substitutions = int(_require_number(group.get("substitutions"), f"{name}.substitutions"))
    deletions = int(_require_number(group.get("deletions"), f"{name}.deletions"))
    insertions = int(_require_number(group.get("insertions"), f"{name}.insertions"))
    reference_words = int(group["reference_words"])
    rate = (substitutions + deletions + insertions) / reference_words
    if not math.isclose(rate, _require_number(group.get("wer"), f"{name}.wer"), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{name} WER derivation mismatch")
    components = sum(
        _require_number(group.get(field), f"{name}.{field}")
        for field in ("asr_latency_ms", "luna_latency_ms", "vad_alignment_latency_ms")
    )
    if not math.isclose(
        components,
        _require_number(group.get("sequential_component_latency_ms"), f"{name}.sequential_component_latency_ms"),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name} component latency derivation mismatch")
    compatible = int(_require_number(group.get("fixed_padding_compatible_files"), f"{name}.fixed"))
    adaptive = int(_require_number(group.get("adaptive_padding_used_files"), f"{name}.adaptive"))
    failed = int(_require_number(group.get("alignment_failed_files"), f"{name}.failed"))
    if compatible + adaptive + failed != int(group["files"]):
        raise ValueError(f"{name} padding coverage mismatch")


def verify(arguments: argparse.Namespace) -> dict[str, object]:
    root = arguments.evidence.resolve()
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    evaluator_path = root / "run-evaluator.py"
    allowed_files = {"manifest.json", "summary.json", "run-evaluator.py"}
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual_files != allowed_files:
        raise ValueError(f"unexpected evidence files: {sorted(actual_files - allowed_files)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest.get("scope") != (
        "privacy-safe aggregate evidence plus exact historic run evaluator; no raw media/content"
    ):
        raise ValueError("evidence scope is not privacy-safe aggregate plus evaluator only")
    files = manifest.get("files")
    if not isinstance(files, list) or {item.get("path") for item in files if isinstance(item, dict)} != {
        "summary.json",
        "run-evaluator.py",
    }:
        raise ValueError("evidence manifest must contain summary and exact run evaluator only")
    files_by_path = {str(item["path"]): item for item in files if isinstance(item, dict)}
    for name, path in (("summary.json", summary_path), ("run-evaluator.py", evaluator_path)):
        item = files_by_path[name]
        if item.get("sha256") != _sha256(path) or item.get("bytes") != path.stat().st_size:
            raise ValueError(f"{name} manifest binding mismatch")
    if _sha256(evaluator_path) != RUN_EVALUATOR_SHA256:
        raise ValueError("exact run evaluator hash mismatch")
    if summary.get("status") != "completed" or summary.get("kind") != "privacy-safe-external-speech-benchmark-summary":
        raise ValueError("summary status/schema is invalid")
    local_run = summary.get("local_run")
    if not isinstance(local_run, dict):
        raise ValueError("local run provenance missing")
    if local_run.get("evaluator_sha256") != RUN_EVALUATOR_SHA256:
        raise ValueError("summary evaluator provenance mismatch")
    if local_run.get("source_manifest_sha256") != RUN_SOURCE_MANIFEST_SHA256:
        raise ValueError("summary source manifest provenance mismatch")
    configuration = local_run.get("configuration")
    if not isinstance(configuration, dict) or set(configuration) != {
        "vad_noise_db",
        "vad_minimum_silence_seconds",
        "speech_start_padding_seconds",
        "speech_end_padding_seconds",
    }:
        raise ValueError("historic run configuration changed")
    groups = summary.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(EXPECTED_GROUPS):
        raise ValueError("source groups changed")
    for name in EXPECTED_GROUPS:
        _validate_group(name, groups[name])
    overall = summary.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("overall aggregate missing")
    group_values = [groups[name] for name in EXPECTED_GROUPS]
    for field in (
        "files",
        "substitutions",
        "deletions",
        "insertions",
        "reference_words",
        "fixed_padding_compatible_files",
        "adaptive_padding_used_files",
        "alignment_failed_files",
    ):
        if overall.get(field) != sum(int(group[field]) for group in group_values):
            raise ValueError(f"overall {field} mismatch")
    for field in (
        "audio_seconds",
        "asr_latency_ms",
        "luna_latency_ms",
        "vad_alignment_latency_ms",
        "sequential_component_latency_ms",
    ):
        if not math.isclose(
            _require_number(overall.get(field), f"overall.{field}"),
            sum(_require_number(group[field], f"group.{field}") for group in group_values),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"overall {field} mismatch")
    errors = int(overall["substitutions"]) + int(overall["deletions"]) + int(overall["insertions"])
    if not math.isclose(
        errors / int(overall["reference_words"]),
        _require_number(overall.get("wer"), "overall.wer"),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("overall WER derivation mismatch")
    wall_ms = _require_number(overall.get("wall_latency_ms"), "overall.wall_latency_ms")
    duration = _require_number(overall.get("audio_seconds"), "overall.audio_seconds")
    if not math.isclose(
        wall_ms / 1000 / duration,
        _require_number(overall.get("real_time_factor"), "overall.real_time_factor"),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("real-time factor derivation mismatch")
    claims = summary.get("claims")
    if not isinstance(claims, dict) or "not measurable" not in str(claims.get("role_accuracy_or_der")):
        raise ValueError("missing DER/role claim boundary")
    return {
        "status": "completed",
        "files": overall["files"],
        "audio_seconds": duration,
        "micro_wer": overall["wer"],
        "fixed_padding_compatible_files": overall["fixed_padding_compatible_files"],
        "adaptive_padding_used_files": overall["adaptive_padding_used_files"],
        "summary_sha256": _sha256(summary_path),
        "run_evaluator_sha256": _sha256(evaluator_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "release-evidence" / "external-speech-2026-08-01",
    )
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
