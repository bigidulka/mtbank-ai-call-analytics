#!/usr/bin/env python3
"""Verify privacy-safe Luna/Terra/Sol aggregate comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
EXPECTED_PROVENANCE = {
    "comparison_evaluator_sha256": "513276f7323c026d93d746fc83ad77b60dfdfff13e5ec8e29f50f2603747ba76",
    "frozen_input_sha256": "49acd880c14f3f2513e8e5b9fe4dfc3b0b69db380635e791161545dfab1c01c0",
    "prompt_hash": "5561ad4216dc28186270ae94edb714b674f25306365799e316ddc27f6a5aa645",
    "comparison_sha256": "661e32648ffe74b269a18f09428931541cd7ff45ec473710812406fda87efa36",
    "scored_sha256": "336c8dfc3be7b235142aa0714a524abfee35703c1886e2565679bcc075c30e0b",
    "references_sha256": "d7e0300b3f3bff5c78415643c28cb2d61f6f77796bd639e888d39706a9c43d1f",
    "score_evaluator_sha256": "418c1c80e096893ace9a7f3c4d8e94b19f3d14c8d78739cdfd03862592f4e368",
    "annotation_boundary": (
        "references cryptographically bind frozen comparison; chronology and annotator blinding remain author-attested"
    ),
}
EXPECTED_MODEL_VALUES = {
    "gpt-5.6-luna": {
        "request_attempts": 36,
        "retried_runs": 0,
        "stable_files": 10,
        "identical_repeat_pairs": 30,
        "median_latency_ms": 5697.3935,
        "mean_latency_ms": 12790.918611111112,
        "role_accuracy_mean": 0.3946424666299327,
        "role_accuracy_stdev": 0.1030362660760209,
        "role_der_mean": 0.6071659880626239,
        "speaker_attributed_wer_mean": 0.315693857365748,
    },
    "gpt-5.6-terra": {
        "request_attempts": 38,
        "retried_runs": 2,
        "stable_files": 8,
        "identical_repeat_pairs": 26,
        "median_latency_ms": 5189.780000000001,
        "mean_latency_ms": 13201.876916666668,
        "role_accuracy_mean": 0.46755391960383513,
        "role_accuracy_stdev": 0.1060003257311632,
        "role_der_mean": 0.5342545350887214,
        "speaker_attributed_wer_mean": 0.2778123557179475,
    },
    "gpt-5.6-sol": {
        "request_attempts": 36,
        "retried_runs": 0,
        "stable_files": 7,
        "identical_repeat_pairs": 24,
        "median_latency_ms": 5557.429,
        "mean_latency_ms": 12245.184611111112,
        "role_accuracy_mean": 0.5013765269560669,
        "role_accuracy_stdev": 0.06931084118242095,
        "role_der_mean": 0.5004319277364896,
        "speaker_attributed_wer_mean": 0.26862905183805386,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def verify(arguments: argparse.Namespace) -> dict[str, object]:
    root = arguments.evidence.resolve()
    allowed = {"manifest.json", "summary.json"}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != allowed:
        raise ValueError(f"unexpected evidence files: {sorted(actual - allowed)}")
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "kind", "scope", "files", "verification_command", "claim_boundary"}:
        raise ValueError("manifest schema is not exact")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "semantic-role-model-comparison-evidence-manifest"
        or manifest.get("scope") != "privacy-safe aggregate evidence only"
        or manifest.get("verification_command") != "uv run python scripts/verify_semantic_model_comparison.py"
        or manifest.get("claim_boundary")
        != (
            "No raw external transcripts, assignments, references, names or provider observations. Sol is best "
            "observed "
            "on two post-run manually role-annotated calls, not an established production winner."
        )
    ):
        raise ValueError("manifest privacy/claim contract changed")
    listed = manifest.get("files")
    if not isinstance(listed, list) or len(listed) != 1 or listed[0].get("path") != "summary.json":
        raise ValueError("manifest must bind summary only")
    if listed[0].get("sha256") != _sha256(summary_path) or listed[0].get("bytes") != summary_path.stat().st_size:
        raise ValueError("summary binding mismatch")
    if set(summary) != {
        "schema_version",
        "kind",
        "status",
        "scope",
        "provenance",
        "design",
        "model_summary",
        "within_model_stability",
        "within_model_disagreement",
        "between_model",
        "post_run_manual_reference_score",
        "claims",
    }:
        raise ValueError("summary schema is not exact")
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "completed"
        or summary.get("kind") != "privacy-safe-semantic-role-model-comparison-summary"
        or summary.get("scope")
        != "aggregate-only; external transcripts, role assignments, names, and model observations excluded"
    ):
        raise ValueError("summary schema/status/privacy scope invalid")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("summary provenance missing")
    for field, expected in EXPECTED_PROVENANCE.items():
        if provenance.get(field) != expected:
            raise ValueError(f"provenance mismatch: {field}")
    for field in ("comparison_sha256", "scored_sha256", "references_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid provenance hash: {field}")
    design = summary.get("design")
    if not isinstance(design, dict) or tuple(design.get("models", ())) != MODELS:
        raise ValueError("model coverage changed")
    if (
        design.get("files") != 12
        or design.get("repeats") != 3
        or design.get("successful_runs") != 108
        or design.get("request_attempts") != 110
        or design.get("retried_runs") != 2
    ):
        raise ValueError("paired design coverage changed")
    if design.get("schedule_seed") != 20260802 or design.get("schedule_strategy") != (
        "seeded randomized model/file/repeat order"
    ):
        raise ValueError("randomized schedule evidence changed")
    if design.get("temperature") != 0 or design.get("references_used_in_inference") is not False:
        raise ValueError("inference isolation changed")
    model_summary = summary.get("model_summary")
    stability = summary.get("within_model_stability")
    within_disagreement = summary.get("within_model_disagreement")
    scored = summary.get("post_run_manual_reference_score")
    if not all(
        isinstance(value, dict) and set(value) == set(MODELS)
        for value in (model_summary, stability, within_disagreement, scored)
    ):
        raise ValueError("model summaries malformed")
    assert (
        isinstance(model_summary, dict)
        and isinstance(stability, dict)
        and isinstance(within_disagreement, dict)
        and isinstance(scored, dict)
    )
    for model in MODELS:
        current = model_summary[model]
        current_stability = stability[model]
        current_disagreement = within_disagreement[model]
        current_score = scored[model]
        expected_counts = EXPECTED_MODEL_VALUES[model]
        if (
            current.get("runs") != 36
            or current.get("request_attempts") != expected_counts["request_attempts"]
            or current.get("retried_runs") != expected_counts["retried_runs"]
            or current_stability.get("files") != 12
        ):
            raise ValueError(f"{model} coverage/request mismatch")
        for field in ("median_latency_ms", "mean_latency_ms"):
            value = _number(current.get(field), f"{model}.{field}")
            if not math.isclose(value, float(expected_counts[field]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{model}.{field} changed")
        if current.get("fixed_alignment_completed") != 30 or current.get("adaptive_alignment_completed") != 36:
            raise ValueError(f"{model} alignment counts mismatch")
        stable_files = current_stability.get("stable_files")
        if (
            not isinstance(stable_files, int)
            or isinstance(stable_files, bool)
            or stable_files != expected_counts["stable_files"]
        ):
            raise ValueError(f"{model} stability count invalid")
        if not math.isclose(
            stable_files / 12,
            _number(current_stability.get("stability"), f"{model}.stability"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{model} stability derivation mismatch")
        if (
            current_disagreement.get("repeat_pairs") != 36
            or current_disagreement.get("identical_repeat_pairs") != expected_counts["identical_repeat_pairs"]
        ):
            raise ValueError(f"{model} repeat-pair coverage mismatch")
        for field in ("mean_word_role_disagreement", "mean_real_call_word_role_disagreement"):
            value = _number(current_disagreement.get(field), f"{model}.{field}")
            if value < 0 or value > 1:
                raise ValueError(f"{model}.{field} out of range")
        for field in (
            "role_accuracy_mean",
            "role_accuracy_stdev",
            "role_der_mean",
            "speaker_attributed_wer_mean",
        ):
            value = _number(current_score.get(field), f"{model}.{field}")
            if not math.isclose(value, float(expected_counts[field]), rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"{model}.{field} changed")
    between = summary.get("between_model")
    if not isinstance(between, dict) or between.get("file_repeats") != 36:
        raise ValueError("between-model coverage mismatch")
    identical = between.get("all_models_identical_file_repeats")
    if not isinstance(identical, int) or isinstance(identical, bool) or identical != 24:
        raise ValueError("identical count invalid")
    pairwise = between.get("pairwise")
    if not isinstance(pairwise, dict) or len(pairwise) != 3:
        raise ValueError("pairwise comparison coverage mismatch")
    for name, value in pairwise.items():
        if not isinstance(value, dict) or value.get("file_repeats") != 36 or value.get("identical_file_repeats") != 26:
            raise ValueError(f"pairwise coverage mismatch: {name}")
        for field in ("mean_word_role_disagreement", "mean_real_call_word_role_disagreement"):
            metric = _number(value.get(field), f"{name}.{field}")
            if metric < 0 or metric > 1:
                raise ValueError(f"pairwise metric out of range: {name}.{field}")
    if not math.isclose(
        _number(between.get("mean_three_model_word_role_disagreement"), "between.mean_disagreement"),
        0.11589189015527326,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("between-model mean disagreement changed")
    claims = summary.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("claims missing")
    if claims.get("model_dependence") != (
        "randomized paired run shows model-associated differences beyond frozen ASR/VAD input; same-model "
        "nondeterminism remains a competing source of variation"
    ):
        raise ValueError("model-dependence claim mismatch")
    if claims.get("best_observed_model") != "gpt-5.6-sol on two manually post-annotated real calls":
        raise ValueError("best observed model claim mismatch")
    if claims.get("production_winner") != (
        "not established: only two calls, source timestamps/transcripts are imperfect, roles were not double-annotated"
    ):
        raise ValueError("production claim boundary missing")
    if claims.get("fixed_padding") != (
        "all models failed fixed 0.2/0.9 padding on both real calls; model choice does not fix core alignment topology"
    ):
        raise ValueError("fixed-padding claim mismatch")
    sol_accuracy = _number(scored["gpt-5.6-sol"].get("role_accuracy_mean"), "sol.role_accuracy")
    if sol_accuracy <= max(
        _number(scored["gpt-5.6-luna"].get("role_accuracy_mean"), "luna.role_accuracy"),
        _number(scored["gpt-5.6-terra"].get("role_accuracy_mean"), "terra.role_accuracy"),
    ):
        raise ValueError("Sol-best arithmetic mismatch")
    return {
        "status": "completed",
        "successful_runs": 108,
        "request_attempts": 110,
        "identical_file_repeats": identical,
        "mean_word_role_disagreement": between["mean_three_model_word_role_disagreement"],
        "best_observed_model": claims["best_observed_model"],
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "release-evidence" / "semantic-role-model-comparison-2026-08-02",
    )
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
