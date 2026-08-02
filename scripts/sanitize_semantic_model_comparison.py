#!/usr/bin/env python3
"""Create privacy-safe aggregate evidence for semantic role model comparison."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(comparison_path: Path, scored_path: Path) -> dict[str, object]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    scored = json.loads(scored_path.read_text(encoding="utf-8"))
    if comparison.get("status") != "completed" or scored.get("status") != "completed":
        raise ValueError("comparison/scored result is incomplete")
    if scored["provenance"]["comparison_sha256"] != _sha256(comparison_path):
        raise ValueError("scored comparison hash mismatch")
    models = tuple(str(item) for item in comparison["models"])
    if models != tuple(str(item) for item in scored["models"]):
        raise ValueError("model coverage mismatch")
    paired = comparison["paired"]
    pairwise: dict[str, object] = {}
    within_model_disagreement: dict[str, object] = {}
    runs = comparison["runs"]
    if not isinstance(paired, list) or not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise ValueError("comparison paired/runs are malformed")

    def word_roles(run: dict[str, object]) -> tuple[str, ...]:
        input_words = run["input_words"]
        assignments = run["assignments"]
        if not isinstance(input_words, int) or isinstance(input_words, bool) or not isinstance(assignments, list):
            raise ValueError("run word/assignment fields are malformed")
        roles = [""] * input_words
        for assignment in assignments:
            if not isinstance(assignment, dict):
                raise ValueError("run assignment is malformed")
            start = assignment["start_word_index"]
            end = assignment["end_word_index"]
            role = assignment["role"]
            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(role, str):
                raise ValueError("run assignment values are malformed")
            for index in range(start, end + 1):
                roles[index] = role
        if any(not role for role in roles):
            raise ValueError("role coverage incomplete")
        return tuple(roles)

    for model in models:
        disagreements: list[float] = []
        real_call_disagreements: list[float] = []
        identical = 0
        for identifier in sorted({str(run["id"]) for run in runs}):
            model_runs = sorted(
                [run for run in runs if run["model"] == model and run["id"] == identifier],
                key=lambda run: int(run["repeat"]),
            )
            for left_run, right_run in itertools.combinations(model_runs, 2):
                left_roles = word_roles(left_run)
                right_roles = word_roles(right_run)
                disagreement = sum(a != b for a, b in zip(left_roles, right_roles, strict=True)) / len(left_roles)
                disagreements.append(disagreement)
                identical += int(disagreement == 0)
                if identifier.startswith("axon-"):
                    real_call_disagreements.append(disagreement)
        within_model_disagreement[model] = {
            "mean_word_role_disagreement": statistics.fmean(disagreements),
            "mean_real_call_word_role_disagreement": statistics.fmean(real_call_disagreements),
            "identical_repeat_pairs": identical,
            "repeat_pairs": len(disagreements),
        }

    for left, right in itertools.combinations(models, 2):
        disagreements: list[float] = []
        call_disagreements: list[float] = []
        identical = 0
        repeats = comparison["repeats"]
        if not isinstance(repeats, int) or isinstance(repeats, bool):
            raise ValueError("comparison repeats is malformed")
        for repeat in range(repeats):
            for identifier in sorted({str(run["id"]) for run in runs}):
                left_run = next(
                    run for run in runs if run["model"] == left and run["repeat"] == repeat and run["id"] == identifier
                )
                right_run = next(
                    run for run in runs if run["model"] == right and run["repeat"] == repeat and run["id"] == identifier
                )
                left_roles = word_roles(left_run)
                right_roles = word_roles(right_run)
                disagreement = sum(a != b for a, b in zip(left_roles, right_roles, strict=True)) / len(left_roles)
                disagreements.append(disagreement)
                identical += int(disagreement == 0)
                if identifier.startswith("axon-"):
                    call_disagreements.append(disagreement)
        pairwise[f"{left}__{right}"] = {
            "mean_word_role_disagreement": statistics.fmean(disagreements),
            "mean_real_call_word_role_disagreement": statistics.fmean(call_disagreements),
            "identical_file_repeats": identical,
            "file_repeats": len(disagreements),
        }
    scored_summary: dict[str, object] = {}
    for model in models:
        summary = scored["summary"][model]
        scored_summary[model] = {
            "role_accuracy_mean": summary["role_accuracy_mean"],
            "role_accuracy_stdev": summary["role_accuracy_stdev"],
            "role_der_mean": summary["role_der_mean"],
            "speaker_attributed_wer_mean": summary["speaker_attributed_wer_mean"],
        }
    return {
        "schema_version": 1,
        "kind": "privacy-safe-semantic-role-model-comparison-summary",
        "status": "completed",
        "scope": "aggregate-only; external transcripts, role assignments, names, and model observations excluded",
        "provenance": {
            "comparison_sha256": _sha256(comparison_path),
            "comparison_evaluator_sha256": comparison["provenance"]["evaluator_sha256"],
            "frozen_input_sha256": comparison["provenance"]["input_sha256"],
            "prompt_hash": comparison["provenance"]["prompt_hash"],
            "scored_sha256": _sha256(scored_path),
            "score_evaluator_sha256": scored["provenance"]["evaluator_sha256"],
            "references_sha256": scored["provenance"]["references_sha256"],
            "annotation_boundary": scored["provenance"]["annotation_boundary"],
        },
        "design": {
            "models": models,
            "files": comparison["files"],
            "repeats": comparison["repeats"],
            "successful_runs": len(runs),
            "request_attempts": sum(int(run["attempts"]) for run in runs),
            "retried_runs": sum(int(run["attempts"]) > 1 for run in runs),
            "temperature": 0,
            "identical_frozen_inputs": True,
            "references_used_in_inference": False,
            "schedule_seed": comparison["provenance"]["schedule_seed"],
            "schedule_strategy": comparison["provenance"]["schedule_strategy"],
        },
        "model_summary": comparison["model_summary"],
        "within_model_stability": comparison["within_model"],
        "within_model_disagreement": within_model_disagreement,
        "between_model": {
            "all_models_identical_file_repeats": sum(item["all_role_boundaries_identical"] for item in paired),
            "file_repeats": len(paired),
            "mean_three_model_word_role_disagreement": statistics.fmean(
                float(item["mean_pairwise_word_role_disagreement"]) for item in paired
            ),
            "median_three_model_word_role_disagreement": statistics.median(
                float(item["mean_pairwise_word_role_disagreement"]) for item in paired
            ),
            "pairwise": pairwise,
        },
        "post_run_manual_reference_score": scored_summary,
        "claims": {
            "model_dependence": (
                "randomized paired run shows model-associated differences beyond frozen ASR/VAD input; same-model "
                "nondeterminism remains a competing source of variation"
            ),
            "best_observed_model": "gpt-5.6-sol on two manually post-annotated real calls",
            "production_winner": (
                "not established: only two calls, source timestamps/transcripts are imperfect, roles were not "
                "double-annotated"
            ),
            "fixed_padding": (
                "all models failed fixed 0.2/0.9 padding on both real calls; model choice does not fix core alignment "
                "topology"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = sanitize(arguments.comparison, arguments.scored)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
