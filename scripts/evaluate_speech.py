#!/usr/bin/env python3
"""Deterministic WER/DER/role metrics for licensed speech-reference evaluation data."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

from mtbank_ai.speech.dataset import ManifestError, validate_manifest

_PUBLIC_ROLES = {"Оператор", "Клиент"}
_TOKEN_BOUNDARY = re.compile(r"[^\w]+", flags=re.UNICODE)


@dataclass(frozen=True)
class Segment:
    identifier: str
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


def normalize_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(token for token in _TOKEN_BOUNDARY.sub(" ", normalized).split() if token)


def word_error_counts(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> ErrorCounts:
    matrix: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hypothesis) + 1)] for _ in range(len(reference) + 1)
    ]
    for row in range(1, len(reference) + 1):
        matrix[row][0] = (row, 0, row, 0)
    for column in range(1, len(hypothesis) + 1):
        matrix[0][column] = (column, 0, 0, column)
    for row, reference_word in enumerate(reference, start=1):
        for column, hypothesis_word in enumerate(hypothesis, start=1):
            if reference_word == hypothesis_word:
                matrix[row][column] = matrix[row - 1][column - 1]
                continue
            candidates = (
                _increment(matrix[row - 1][column - 1], substitutions=1),
                _increment(matrix[row - 1][column], deletions=1),
                _increment(matrix[row][column - 1], insertions=1),
            )
            matrix[row][column] = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    _, substitutions, deletions, insertions = matrix[-1][-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def corpus_wer(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> ErrorCounts:
    """Compares time-ordered text, not evaluator-specific segment IDs or boundaries."""

    reference_tokens = tuple(
        token
        for segment in _time_ordered(reference)
        for token in normalize_tokens(segment.text)
    )
    hypothesis_tokens = tuple(
        token
        for segment in _time_ordered(hypothesis)
        for token in normalize_tokens(segment.text)
    )
    return word_error_counts(reference_tokens, hypothesis_tokens)


def _time_ordered(segments: tuple[Segment, ...]) -> tuple[Segment, ...]:
    return tuple(sorted(segments, key=lambda segment: (segment.start, segment.end, segment.identifier)))


def diarization_error_rate(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> dict[str, float]:
    boundaries = _boundaries(reference, hypothesis)
    if len(boundaries) < 2:
        raise ValueError("DER требует хотя бы один reference speech interval")
    mapping = _optimal_speaker_mapping(reference, hypothesis, boundaries)
    reference_time = miss = false_alarm = confusion = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        duration = right - left
        midpoint = left + duration / 2
        reference_speakers = _active_speakers(reference, midpoint)
        hypothesis_speakers = _active_speakers(hypothesis, midpoint)
        reference_time += len(reference_speakers) * duration
        if not reference_speakers:
            false_alarm += len(hypothesis_speakers) * duration
            continue
        if not hypothesis_speakers:
            miss += len(reference_speakers) * duration
            continue
        for speaker in reference_speakers:
            if any(mapping.get(hypothesis_speaker) == speaker for hypothesis_speaker in hypothesis_speakers):
                continue
            confusion += duration
        false_alarm += max(0, len(hypothesis_speakers) - len(reference_speakers)) * duration
    if reference_time == 0:
        raise ValueError("DER requires positive reference speech time")
    return {
        "der": (miss + false_alarm + confusion) / reference_time,
        "miss_seconds": miss,
        "false_alarm_seconds": false_alarm,
        "confusion_seconds": confusion,
        "reference_speaker_seconds": reference_time,
    }


def time_weighted_role_accuracy(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> float:
    _require_public_roles(reference, hypothesis)
    boundaries = _boundaries(reference, hypothesis)
    total = correct = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        duration = right - left
        midpoint = left + duration / 2
        reference_roles = _active_speakers(reference, midpoint)
        hypothesis_roles = _active_speakers(hypothesis, midpoint)
        total += len(reference_roles) * duration
        correct += len(reference_roles & hypothesis_roles) * duration
    if total == 0:
        raise ValueError("role accuracy requires positive reference speech time")
    return correct / total


def speaker_attributed_wer(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> ErrorCounts:
    _require_public_roles(reference, hypothesis)
    counts = ErrorCounts(0, 0, 0, 0)
    for role in sorted(_PUBLIC_ROLES):
        reference_tokens = tuple(
            token
            for segment in reference
            if segment.speaker == role
            for token in normalize_tokens(segment.text)
        )
        hypothesis_tokens = tuple(
            token
            for segment in hypothesis
            if segment.speaker == role
            for token in normalize_tokens(segment.text)
        )
        counts = _add_counts(counts, word_error_counts(reference_tokens, hypothesis_tokens))
    return counts


def load_segments(path: Path) -> tuple[Segment, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reference/hypothesis JSON is unreadable") from error
    records = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("reference/hypothesis JSON requires segments list")
    segments = tuple(_parse_segment(record) for record in records)
    identifiers = tuple(segment.identifier for segment in segments)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("segment IDs must be unique")
    return segments


def _parse_segment(raw: Any) -> Segment:
    if not isinstance(raw, dict) or set(raw) != {"id", "start", "end", "speaker", "text"}:
        raise ValueError("segment schema must be exact")
    identifier, speaker, text = raw["id"], raw["speaker"], raw["text"]
    if not all(isinstance(value, str) and value.strip() for value in (identifier, speaker, text)):
        raise ValueError("segment id/speaker/text must be non-empty strings")
    start, end = float(raw["start"]), float(raw["end"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
        raise ValueError("segment timestamps must be finite and increasing")
    return Segment(identifier, start, end, speaker, text)


def _increment(counts: tuple[int, int, int, int], **changes: int) -> tuple[int, int, int, int]:
    cost, substitutions, deletions, insertions = counts
    substitution = changes.get("substitutions", 0)
    deletion = changes.get("deletions", 0)
    insertion = changes.get("insertions", 0)
    return (
        cost + substitution + deletion + insertion,
        substitutions + substitution,
        deletions + deletion,
        insertions + insertion,
    )


def _add_counts(left: ErrorCounts, right: ErrorCounts) -> ErrorCounts:
    return ErrorCounts(
        substitutions=left.substitutions + right.substitutions,
        deletions=left.deletions + right.deletions,
        insertions=left.insertions + right.insertions,
        reference_words=left.reference_words + right.reference_words,
    )


def _boundaries(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> tuple[float, ...]:
    boundaries = sorted({time for segment in (*reference, *hypothesis) for time in (segment.start, segment.end)})
    return tuple(boundaries)


def _active_speakers(segments: tuple[Segment, ...], midpoint: float) -> set[str]:
    return {segment.speaker for segment in segments if segment.start <= midpoint < segment.end}


def _optimal_speaker_mapping(
    reference: tuple[Segment, ...],
    hypothesis: tuple[Segment, ...],
    boundaries: tuple[float, ...],
) -> dict[str, str | None]:
    reference_labels = sorted({segment.speaker for segment in reference})
    hypothesis_labels = sorted({segment.speaker for segment in hypothesis})
    if len(reference_labels) > 8 or len(hypothesis_labels) > 8:
        raise ValueError("DER supports at most eight speaker labels per recording")
    overlap = {
        (reference_label, hypothesis_label): 0.0
        for reference_label in reference_labels
        for hypothesis_label in hypothesis_labels
    }
    for left, right in zip(boundaries, boundaries[1:]):
        midpoint = (left + right) / 2
        for reference_label in _active_speakers(reference, midpoint):
            for hypothesis_label in _active_speakers(hypothesis, midpoint):
                overlap[reference_label, hypothesis_label] += right - left

    padded_reference = tuple(reference_labels) + (None,) * max(0, len(hypothesis_labels) - len(reference_labels))
    best_mapping: dict[str, str | None] = {}
    best_overlap = -1.0
    for assignment in permutations(padded_reference, len(hypothesis_labels)):
        score = sum(
            overlap.get((reference_label, hypothesis_label), 0.0)
            for hypothesis_label, reference_label in zip(hypothesis_labels, assignment, strict=True)
            if reference_label is not None
        )
        if score > best_overlap:
            best_overlap = score
            best_mapping = dict(zip(hypothesis_labels, assignment, strict=True))
    return best_mapping


def _require_public_roles(reference: tuple[Segment, ...], hypothesis: tuple[Segment, ...]) -> None:
    labels = {segment.speaker for segment in (*reference, *hypothesis)}
    if not labels.issubset(_PUBLIC_ROLES):
        raise ValueError("role metrics accept only exact public roles Оператор and Клиент")


def _counts_json(counts: ErrorCounts) -> dict[str, float | int]:
    return {
        "wer": counts.rate,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "reference_words": counts.reference_words,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("test_data/manifest.yaml"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        validate_manifest(arguments.manifest, require_release_corpus=True)
        reference = load_segments(arguments.reference)
        hypothesis = load_segments(arguments.hypothesis)
        result = {
            "wer": _counts_json(corpus_wer(reference, hypothesis)),
            "der": diarization_error_rate(reference, hypothesis),
            "time_weighted_role_accuracy": time_weighted_role_accuracy(reference, hypothesis),
            "speaker_attributed_wer": _counts_json(speaker_attributed_wer(reference, hypothesis)),
        }
    except (ManifestError, ValueError) as error:
        print(f"evaluation blocked: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
