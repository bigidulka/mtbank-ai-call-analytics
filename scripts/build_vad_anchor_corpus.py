#!/usr/bin/env python3
"""Build speech-only VAD anchors for flat synthetic transcription experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import cast

from mtbank_ai.speech.dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
_EVENT = re.compile(r"silence_(start|end):\s*([0-9.]+)")


def _anchors(audio: Path, duration: float, *, noise_db: float, minimum_silence: float) -> list[dict[str, float | int]]:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={minimum_silence}",
            "-f",
            "null",
            "-",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=120,
    )
    if process.returncode != 0:
        raise ValueError(f"ffmpeg VAD failed for {audio.name}")
    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for kind, raw_value in _EVENT.findall(process.stderr):
        value = float(raw_value)
        if kind == "start":
            pending = value
        elif pending is not None:
            silences.append((pending, value))
            pending = None
    if pending is not None:
        silences.append((pending, duration))

    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silences:
        start = min(max(start, cursor), duration)
        end = min(max(end, start), duration)
        if start - cursor >= 0.03:
            speech.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= 0.03:
        speech.append((cursor, duration))
    if not speech:
        raise ValueError(f"VAD produced no speech anchors for {audio.name}")
    return [
        {"anchor_index": index, "start": round(start, 6), "end": round(end, 6)}
        for index, (start, end) in enumerate(speech)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(arguments: argparse.Namespace) -> dict[str, object]:
    transcriptions = json.loads(arguments.transcriptions.read_text(encoding="utf-8"))
    if not isinstance(transcriptions, dict) or transcriptions.get("status") != "completed":
        raise ValueError("flat transcription corpus is not completed")
    raw_files = transcriptions.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("flat transcription corpus is invalid")
    entries = tuple(
        entry
        for entry in validate_manifest(arguments.manifest, require_release_corpus=True)
        if entry.kind == "speech_reference"
    )
    expected_ids = tuple(entry.identifier for entry in entries)
    actual_ids = tuple(str(item.get("id")) for item in raw_files if isinstance(item, dict))
    if actual_ids != expected_ids or len(actual_ids) != len(raw_files) or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("flat transcription corpus must exactly cover manifest entries in order")
    transcriptions_by_id = {str(item["id"]): cast(dict[str, object], item) for item in raw_files}
    files: list[dict[str, object]] = []
    for entry in entries:
        transcription = transcriptions_by_id.get(entry.identifier)
        if transcription is None:
            raise ValueError(f"missing flat transcription for {entry.identifier}")
        duration = float(entry.raw["duration_seconds"])
        files.append(
            {
                "id": entry.identifier,
                "audio_sha256": entry.raw["sha256"],
                "duration_seconds": duration,
                "asr_latency_ms": transcription["latency_ms"],
                "text": transcription["text"],
                "anchors": _anchors(
                    entry.path,
                    duration,
                    noise_db=arguments.noise_db,
                    minimum_silence=arguments.minimum_silence,
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "flat-transcript-vad-anchor-corpus",
        "status": "completed",
        "scope": "approved synthetic/no-PII",
        "provenance": {
            "manifest_path": str(arguments.manifest),
            "manifest_sha256": _sha256(arguments.manifest),
            "transcriptions_path": str(arguments.transcriptions),
            "transcriptions_sha256": _sha256(arguments.transcriptions),
        },
        "vad": {
            "implementation": "ffmpeg silencedetect speech-complement",
            "noise_db": arguments.noise_db,
            "minimum_silence_seconds": arguments.minimum_silence,
            "speaker_information": False,
            "word_information": False,
        },
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--transcriptions", type=Path, required=True)
    parser.add_argument("--noise-db", type=float, default=-45.0)
    parser.add_argument("--minimum-silence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = build(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "flat-transcript-vad-anchor-corpus",
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
