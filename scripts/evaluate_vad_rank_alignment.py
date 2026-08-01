#!/usr/bin/env python3
"""Align semantic turns to VAD anchors using longest-gap boundary ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

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

from mtbank_ai.domain.base import StrictFrozenModel
from mtbank_ai.speech.dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
METRIC_EVALUATOR = ROOT / "scripts" / "evaluate_speech.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SemanticAssignment(StrictFrozenModel):
    start_word_index: int = Field(ge=0, le=1_200)
    end_word_index: int = Field(ge=0, le=1_200)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def valid_range(self) -> SemanticAssignment:
        if self.start_word_index > self.end_word_index:
            raise ValueError("semantic assignment range is reversed")
        return self


class RoleFile(StrictFrozenModel):
    id: str = Field(min_length=1, max_length=256)
    assignments: tuple[SemanticAssignment, ...] = Field(min_length=1, max_length=1_200)


class Anchor(StrictFrozenModel):
    anchor_index: int = Field(ge=0, le=512)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_interval(self) -> Anchor:
        if self.start >= self.end:
            raise ValueError("anchor interval is reversed")
        return self


def _reference(manifest: Path, identifier: str) -> tuple[Segment, ...]:
    entry = next(
        item
        for item in validate_manifest(manifest, require_release_corpus=True)
        if item.identifier == identifier and item.kind == "speech_reference"
    )
    raw = json.loads((entry.root / str(entry.raw["reference_path"])).read_text())["segments"]
    return tuple(Segment(str(x["id"]), x["start"], x["end"], x["speaker"], x["text"]) for x in raw)


def _validate_assignments(assignments: tuple[SemanticAssignment, ...], word_count: int) -> None:
    expected = 0
    for assignment in assignments:
        if assignment.start_word_index != expected:
            raise ValueError("semantic assignments have gap, overlap, or reorder")
        expected = assignment.end_word_index + 1
    if expected != word_count:
        raise ValueError("semantic assignments do not cover all words")


def _merge_same_role(assignments: tuple[SemanticAssignment, ...]) -> tuple[SemanticAssignment, ...]:
    merged: list[SemanticAssignment] = []
    for assignment in assignments:
        if merged and merged[-1].role == assignment.role:
            previous = merged[-1]
            merged[-1] = SemanticAssignment(
                start_word_index=previous.start_word_index,
                end_word_index=assignment.end_word_index,
                role=previous.role,
                confidence=min(previous.confidence, assignment.confidence),
            )
        else:
            merged.append(assignment)
    return tuple(merged)


def _hypothesis(
    source: dict[str, object],
    assignments: tuple[SemanticAssignment, ...],
    *,
    speech_start_padding: float,
    speech_end_padding: float,
) -> tuple[Segment, ...]:
    words = str(source["text"]).split()
    if not words:
        raise ValueError("source transcript is empty")
    duration = source.get("duration_seconds")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("source duration is invalid")
    if (
        not math.isfinite(speech_start_padding)
        or speech_start_padding < 0
        or not math.isfinite(speech_end_padding)
        or speech_end_padding < 0
    ):
        raise ValueError("speech edge padding is invalid")
    _validate_assignments(assignments, len(words))
    raw_anchors = source.get("anchors")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise ValueError("anchors are invalid")
    anchors = tuple(Anchor.model_validate(item, strict=True) for item in raw_anchors)
    if tuple(anchor.anchor_index for anchor in anchors) != tuple(range(len(anchors))):
        raise ValueError("anchor indices must be contiguous")
    if any(
        not math.isfinite(anchor.start) or not math.isfinite(anchor.end) or anchor.end > duration for anchor in anchors
    ):
        raise ValueError("anchor interval is outside source duration")
    if any(current.start < previous.end for previous, current in zip(anchors, anchors[1:])):
        raise ValueError("anchors overlap or are not ordered")
    turns = _merge_same_role(assignments)
    if len(turns) > len(anchors):
        raise ValueError("more semantic turns than VAD anchors")
    gaps = [right.start - left.end for left, right in zip(anchors, anchors[1:])]
    boundary_indices = sorted(sorted(range(len(gaps)), key=lambda index: (-gaps[index], index))[: len(turns) - 1])
    groups: list[tuple[int, int]] = []
    start = 0
    for boundary in boundary_indices:
        groups.append((start, boundary))
        start = boundary + 1
    groups.append((start, len(anchors) - 1))
    segments = tuple(
        Segment(
            f"rank-{index}",
            max(0.0, anchors[left].start - speech_start_padding),
            min(float(duration), anchors[right].end + speech_end_padding),
            turn.role,
            " ".join(words[turn.start_word_index : turn.end_word_index + 1]),
        )
        for index, (turn, (left, right)) in enumerate(zip(turns, groups, strict=True))
    )
    if any(current.start < previous.end for previous, current in zip(segments, segments[1:])):
        raise ValueError("generated hypothesis intervals overlap or reorder")
    return segments


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    entries = tuple(
        item
        for item in validate_manifest(arguments.manifest, require_release_corpus=True)
        if item.kind == "speech_reference"
    )
    expected_ids = tuple(item.identifier for item in entries)
    raw_corpus = json.loads(arguments.corpus.read_text())
    if not isinstance(raw_corpus, dict) or raw_corpus.get("status") != "completed":
        raise ValueError("VAD corpus is not completed")
    raw_corpus_files = raw_corpus.get("files") if isinstance(raw_corpus, dict) else None
    if not isinstance(raw_corpus_files, list):
        raise ValueError("VAD corpus requires files list")
    corpus_ids = tuple(str(item.get("id")) for item in raw_corpus_files if isinstance(item, dict))
    if corpus_ids != expected_ids or len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("VAD corpus must exactly cover manifest entries in order")
    corpus = {str(item["id"]): item for item in raw_corpus_files}

    raw_roles = json.loads(arguments.roles.read_text())
    if not isinstance(raw_roles, dict) or raw_roles.get("status") != "completed":
        raise ValueError("role corpus is not completed")
    raw_role_files = raw_roles.get("files") if isinstance(raw_roles, dict) else None
    if not isinstance(raw_role_files, list):
        raise ValueError("role corpus requires files list")
    if any(not isinstance(item, dict) for item in raw_role_files):
        raise ValueError("role corpus files must be objects")
    role_files = tuple(
        RoleFile.model_validate({"id": item.get("id"), "assignments": tuple(item.get("assignments", ()))}, strict=True)
        for item in raw_role_files
    )
    role_ids = tuple(item.id for item in role_files)
    if role_ids != expected_ids or len(set(role_ids)) != len(role_ids):
        raise ValueError("role corpus must exactly cover manifest entries in order")

    files = []
    miss = false_alarm = confusion = reference_seconds = correct_seconds = 0.0
    attributed_errors = attributed_words = 0
    wer_substitutions = wer_deletions = wer_insertions = wer_reference_words = 0
    for role_file in role_files:
        identifier = role_file.id
        hypothesis = _hypothesis(
            corpus[identifier],
            role_file.assignments,
            speech_start_padding=arguments.speech_start_padding,
            speech_end_padding=arguments.speech_end_padding,
        )
        reference = _reference(arguments.manifest, identifier)
        wer = corpus_wer(reference, hypothesis)
        der = diarization_error_rate(reference, hypothesis)
        role_accuracy = time_weighted_role_accuracy(reference, hypothesis)
        attributed = speaker_attributed_wer(reference, hypothesis)
        miss += der["miss_seconds"]
        false_alarm += der["false_alarm_seconds"]
        confusion += der["confusion_seconds"]
        current_seconds = der["reference_speaker_seconds"]
        reference_seconds += current_seconds
        correct_seconds += role_accuracy * current_seconds
        attributed_errors += attributed.errors
        attributed_words += attributed.reference_words
        wer_substitutions += wer.substitutions
        wer_deletions += wer.deletions
        wer_insertions += wer.insertions
        wer_reference_words += wer.reference_words
        files.append(
            {
                "id": identifier,
                "audio_sha256": corpus[identifier].get("audio_sha256"),
                "turns": len(hypothesis),
                "segments": [
                    {
                        "id": segment.identifier,
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": segment.speaker,
                        "text": segment.text,
                    }
                    for segment in hypothesis
                ],
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
        "kind": "vad-ranked-gap-semantic-alignment",
        "status": "completed",
        "provenance": {
            "manifest_path": str(arguments.manifest),
            "manifest_sha256": _sha256(arguments.manifest),
            "vad_corpus_path": str(arguments.corpus),
            "vad_corpus_sha256": _sha256(arguments.corpus),
            "roles_path": str(arguments.roles),
            "roles_sha256": _sha256(arguments.roles),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "metric_evaluator_path": str(METRIC_EVALUATOR),
            "metric_evaluator_sha256": _sha256(METRIC_EVALUATOR),
        },
        "claim_boundary": "semantic role turns mapped to VAD groups separated by globally longest speech gaps",
        "speech_start_padding_seconds": arguments.speech_start_padding,
        "speech_end_padding_seconds": arguments.speech_end_padding,
        "files": files,
        "micro": {
            "wer": {
                "wer": (wer_substitutions + wer_deletions + wer_insertions) / wer_reference_words,
                "substitutions": wer_substitutions,
                "deletions": wer_deletions,
                "insertions": wer_insertions,
                "reference_words": wer_reference_words,
            },
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
    parser.add_argument("--speech-start-padding", type=float, default=0.2)
    parser.add_argument("--speech-end-padding", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "vad-ranked-gap-semantic-alignment",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
