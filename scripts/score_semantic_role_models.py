#!/usr/bin/env python3
"""Score paired semantic-role model outputs against post-run external call references."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

if __package__:
    from .evaluate_speech import Segment, speaker_attributed_wer
    from .evaluate_vad_rank_alignment import SemanticAssignment, _hypothesis
else:
    from evaluate_speech import Segment, speaker_attributed_wer
    from evaluate_vad_rank_alignment import SemanticAssignment, _hypothesis


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_references(path: Path, *, expected_comparison_sha256: str) -> dict[str, tuple[Segment, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "completed"
        or payload.get("annotation_protocol")
        != "post-run manual role annotation; model outputs unavailable to annotator"
        or payload.get("frozen_comparison_sha256") != expected_comparison_sha256
    ):
        raise ValueError("reference annotation boundary/binding is invalid")
    result: dict[str, tuple[Segment, ...]] = {}
    for file in payload["files"]:
        identifier = str(file["id"])
        segments = tuple(
            Segment(str(item["id"]), float(item["start"]), float(item["end"]), str(item["speaker"]), str(item["text"]))
            for item in file["segments"]
        )
        if not segments or identifier in result:
            raise ValueError("reference coverage is invalid")
        result[identifier] = segments
    return result


def _role_metrics(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> dict[str, float]:
    boundaries = sorted({value for segment in (*reference, *hypothesis) for value in (segment.start, segment.end)})
    total = correct = miss = false_alarm = confusion = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        duration = right - left
        if duration <= 0:
            continue
        midpoint = left + duration / 2
        reference_roles = {segment.speaker for segment in reference if segment.start <= midpoint < segment.end}
        hypothesis_roles = {segment.speaker for segment in hypothesis if segment.start <= midpoint < segment.end}
        total += len(reference_roles) * duration
        correct += len(reference_roles & hypothesis_roles) * duration
        if reference_roles and not hypothesis_roles:
            miss += len(reference_roles) * duration
        elif hypothesis_roles and not reference_roles:
            false_alarm += len(hypothesis_roles) * duration
        elif reference_roles and hypothesis_roles:
            matched = len(reference_roles & hypothesis_roles)
            missing_roles = max(0, len(reference_roles) - len(hypothesis_roles))
            confused_roles = max(0, min(len(reference_roles), len(hypothesis_roles)) - matched)
            extra_roles = max(0, len(hypothesis_roles) - len(reference_roles))
            miss += missing_roles * duration
            confusion += confused_roles * duration
            false_alarm += extra_roles * duration
    if total <= 0:
        raise ValueError("reference contains no role time")
    return {
        "time_weighted_role_accuracy": correct / total,
        "role_der": (miss + false_alarm + confusion) / total,
        "miss_seconds": miss,
        "false_alarm_seconds": false_alarm,
        "confusion_seconds": confusion,
        "reference_role_seconds": total,
    }


def score(arguments: argparse.Namespace) -> dict[str, object]:
    comparison = json.loads(arguments.comparison.read_text(encoding="utf-8"))
    frozen = json.loads(arguments.input.read_text(encoding="utf-8"))
    comparison_sha256 = _sha256(arguments.comparison)
    references = _load_references(arguments.references, expected_comparison_sha256=comparison_sha256)
    if comparison.get("status") != "completed" or frozen.get("status") != "completed":
        raise ValueError("comparison/input is incomplete")
    if comparison["provenance"]["input_sha256"] != _sha256(arguments.input):
        raise ValueError("comparison/input hash mismatch")
    inputs = {str(item["id"]): item for item in frozen["files"]}
    models = tuple(str(item) for item in comparison["models"])
    scored: list[dict[str, object]] = []
    for run in comparison["runs"]:
        identifier = str(run["id"])
        if identifier not in references:
            continue
        source = inputs[identifier]
        assignments = tuple(SemanticAssignment.model_validate(item, strict=True) for item in run["assignments"])
        hypothesis = _hypothesis(
            {
                "text": source["asr_text"],
                "duration_seconds": source["audio_seconds"],
                "anchors": source["vad_anchors"],
            },
            assignments,
            speech_start_padding=float(run["adaptive_start_padding_seconds"]),
            speech_end_padding=float(run["adaptive_end_padding_seconds"]),
        )
        metrics = _role_metrics(references[identifier], hypothesis)
        attributed = speaker_attributed_wer(references[identifier], hypothesis)
        scored.append(
            {
                "repeat": run["repeat"],
                "model": run["model"],
                "id": identifier,
                **metrics,
                "speaker_attributed_wer": attributed.rate,
                "speaker_attributed_errors": attributed.errors,
                "speaker_attributed_reference_words": attributed.reference_words,
            }
        )

    def _value(item: dict[str, object], field: str) -> float:
        value = item[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"scored {field} is not numeric")
        return float(value)

    summary: dict[str, object] = {}
    for model in models:
        relevant = [item for item in scored if item["model"] == model]
        by_file: dict[str, object] = {}
        for identifier in references:
            file_runs = [item for item in relevant if item["id"] == identifier]
            by_file[identifier] = {
                "role_accuracy_mean": statistics.fmean(
                    _value(item, "time_weighted_role_accuracy") for item in file_runs
                ),
                "role_accuracy_range": [
                    min(_value(item, "time_weighted_role_accuracy") for item in file_runs),
                    max(_value(item, "time_weighted_role_accuracy") for item in file_runs),
                ],
                "role_der_mean": statistics.fmean(_value(item, "role_der") for item in file_runs),
                "speaker_attributed_wer_mean": statistics.fmean(
                    _value(item, "speaker_attributed_wer") for item in file_runs
                ),
            }
        summary[model] = {
            "runs": len(relevant),
            "role_accuracy_mean": statistics.fmean(_value(item, "time_weighted_role_accuracy") for item in relevant),
            "role_accuracy_stdev": statistics.pstdev(_value(item, "time_weighted_role_accuracy") for item in relevant),
            "role_der_mean": statistics.fmean(_value(item, "role_der") for item in relevant),
            "speaker_attributed_wer_mean": statistics.fmean(
                _value(item, "speaker_attributed_wer") for item in relevant
            ),
            "by_file": by_file,
        }
    return {
        "schema_version": 1,
        "kind": "external-call-semantic-role-model-score",
        "status": "completed",
        "provenance": {
            "comparison_sha256": comparison_sha256,
            "input_sha256": _sha256(arguments.input),
            "references_sha256": _sha256(arguments.references),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "annotation_boundary": (
                "references cryptographically bind frozen comparison; chronology and annotator blinding remain "
                "author-attested"
            ),
        },
        "models": models,
        "files": tuple(references),
        "scored": scored,
        "summary": summary,
        "claim_boundary": (
            "Reference timestamps/text originate from source-provided transcripts and manual post-run role annotation; "
            "they are not independently double-annotated acoustic truth. Results support model-dependence analysis, "
            "not "
            "production accuracy claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = score(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
