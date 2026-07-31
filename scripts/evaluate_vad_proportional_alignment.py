#!/usr/bin/env python3
"""Align Luna semantic turns to speaker-agnostic VAD speech time proportionally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

if __package__:
    from .evaluate_speech import (
        Segment,
        _counts_json,
        corpus_wer,
        diarization_error_rate,
        speaker_attributed_wer,
        time_weighted_role_accuracy,
    )
else:
    from evaluate_speech import (
        Segment,
        _counts_json,
        corpus_wer,
        diarization_error_rate,
        speaker_attributed_wer,
        time_weighted_role_accuracy,
    )

from mtbank_ai.speech.dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def _reference(manifest: Path, identifier: str) -> tuple[Segment, ...]:
    entry = next(
        item
        for item in validate_manifest(manifest, require_release_corpus=True)
        if item.identifier == identifier and item.kind == "speech_reference"
    )
    raw = json.loads((entry.root / str(entry.raw["reference_path"])).read_text())["segments"]
    return tuple(Segment(str(x["id"]), x["start"], x["end"], x["speaker"], x["text"]) for x in raw)


def _timeline(anchors: list[dict[str, Any]]) -> tuple[list[tuple[float, float, float]], float]:
    cursor = 0.0
    timeline = []
    for anchor in anchors:
        start, end = float(anchor["start"]), float(anchor["end"])
        timeline.append((cursor, cursor + end - start, start))
        cursor += end - start
    return timeline, cursor


def _at_speech_time(timeline: list[tuple[float, float, float]], value: float) -> float:
    for speech_start, speech_end, audio_start in timeline:
        if value <= speech_end:
            return audio_start + max(0.0, value - speech_start)
    speech_start, speech_end, audio_start = timeline[-1]
    return audio_start + speech_end - speech_start


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    corpus = {str(x["id"]): x for x in json.loads(arguments.corpus.read_text())["files"]}
    roles = json.loads(arguments.roles.read_text())["files"]
    files = []
    miss = false_alarm = confusion = reference_seconds = correct_seconds = 0.0
    attributed_errors = attributed_words = 0
    for role_file in roles:
        identifier = str(role_file["id"])
        source = corpus[identifier]
        words = str(source["text"]).split()
        anchors = cast(list[dict[str, Any]], source["anchors"])
        timeline, speech_duration = _timeline(anchors)
        assignments = role_file["assignments"]
        hypothesis = []
        for index, turn in enumerate(assignments):
            left_ratio = int(turn["start_word_index"]) / len(words)
            right_ratio = (int(turn["end_word_index"]) + 1) / len(words)
            start = _at_speech_time(timeline, speech_duration * left_ratio)
            end = _at_speech_time(timeline, speech_duration * right_ratio)
            hypothesis.append(
                Segment(
                    f"proportional-{index}",
                    start,
                    max(end, start + 0.001),
                    str(turn["role"]),
                    " ".join(words[int(turn["start_word_index"]) : int(turn["end_word_index"]) + 1]),
                )
            )
        reference = _reference(arguments.manifest, identifier)
        hypothesis_tuple = tuple(hypothesis)
        wer = corpus_wer(reference, hypothesis_tuple)
        der = diarization_error_rate(reference, hypothesis_tuple)
        role_accuracy = time_weighted_role_accuracy(reference, hypothesis_tuple)
        attributed = speaker_attributed_wer(reference, hypothesis_tuple)
        miss += der["miss_seconds"]
        false_alarm += der["false_alarm_seconds"]
        confusion += der["confusion_seconds"]
        current_seconds = der["reference_speaker_seconds"]
        reference_seconds += current_seconds
        correct_seconds += role_accuracy * current_seconds
        attributed_errors += attributed.errors
        attributed_words += attributed.reference_words
        files.append(
            {
                "id": identifier,
                "metrics": {
                    "wer": _counts_json(wer),
                    "der": der,
                    "time_weighted_role_accuracy": role_accuracy,
                    "speaker_attributed_wer": _counts_json(attributed),
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "vad-proportional-semantic-alignment",
        "status": "completed",
        "claim_boundary": "approximate turn timestamps from cumulative VAD speech time and flat-text word ratios",
        "files": files,
        "micro": {
            "der": {
                "der": (miss + false_alarm + confusion) / reference_seconds,
                "miss_seconds": miss,
                "false_alarm_seconds": false_alarm,
                "confusion_seconds": confusion,
                "reference_speaker_seconds": reference_seconds,
            },
            "time_weighted_role_accuracy": correct_seconds / reference_seconds,
            "speaker_attributed_wer": attributed_errors / attributed_words,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "vad-proportional-semantic-alignment",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
