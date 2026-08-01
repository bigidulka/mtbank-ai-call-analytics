#!/usr/bin/env python3
"""Reproduce leave-one-call-out selection for ranked VAD alignment parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, cast

from build_vad_anchor_corpus import build
from evaluate_vad_rank_alignment import evaluate

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_vad_anchor_corpus.py"
ALIGNER = ROOT / "scripts" / "evaluate_vad_rank_alignment.py"
METRICS = ROOT / "scripts" / "evaluate_speech.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments: argparse.Namespace) -> dict[str, object]:
    noise_values = tuple(arguments.noise_db)
    start_padding_values = tuple(arguments.speech_start_padding)
    end_padding_values = tuple(arguments.speech_end_padding)
    configurations: list[dict[str, object]] = []
    temp = arguments.output.parent / ".vad-rank-loco"
    temp.mkdir(parents=True, exist_ok=True)
    try:
        for noise_db, start_padding, end_padding in product(noise_values, start_padding_values, end_padding_values):
            corpus = build(
                argparse.Namespace(
                    manifest=arguments.manifest,
                    transcriptions=arguments.transcriptions,
                    noise_db=noise_db,
                    minimum_silence=arguments.minimum_silence,
                )
            )
            corpus_path = temp / f"corpus-{abs(noise_db):g}.json"
            corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
            result = evaluate(
                argparse.Namespace(
                    manifest=arguments.manifest,
                    corpus=corpus_path,
                    roles=arguments.roles,
                    speech_start_padding=start_padding,
                    speech_end_padding=end_padding,
                )
            )
            result_files = cast(list[dict[str, Any]], result["files"])
            file_der = {file["id"]: file["metrics"]["der"]["der"] for file in result_files}
            configurations.append(
                {
                    "noise_db": noise_db,
                    "speech_start_padding": start_padding,
                    "speech_end_padding": end_padding,
                    "file_der": file_der,
                    "macro_der": sum(file_der.values()) / len(file_der),
                }
            )
        ids = tuple(cast(dict[str, float], configurations[0]["file_der"]))
        folds = []
        held_out = []
        for identifier in ids:
            best = min(
                configurations,
                key=lambda config: (
                    sum(value for key, value in cast(dict[str, float], config["file_der"]).items() if key != identifier)
                    / (len(ids) - 1),
                    cast(float, config["noise_db"]),
                    cast(float, config["speech_start_padding"]),
                    cast(float, config["speech_end_padding"]),
                ),
            )
            held = cast(dict[str, float], best["file_der"])[identifier]
            held_out.append(held)
            folds.append(
                {
                    "held_out_id": identifier,
                    "selected_noise_db": best["noise_db"],
                    "selected_speech_start_padding": best["speech_start_padding"],
                    "selected_speech_end_padding": best["speech_end_padding"],
                    "held_out_der": held,
                }
            )
        return {
            "schema_version": 1,
            "kind": "vad-ranked-gap-loco-validation",
            "status": "completed",
            "provenance": {
                "manifest_path": str(arguments.manifest),
                "manifest_sha256": _sha256(arguments.manifest),
                "transcriptions_path": str(arguments.transcriptions),
                "transcriptions_sha256": _sha256(arguments.transcriptions),
                "roles_path": str(arguments.roles),
                "roles_sha256": _sha256(arguments.roles),
                "evaluator_path": str(Path(__file__).resolve()),
                "evaluator_sha256": _sha256(Path(__file__).resolve()),
                "builder_path": str(BUILDER),
                "builder_sha256": _sha256(BUILDER),
                "aligner_path": str(ALIGNER),
                "aligner_sha256": _sha256(ALIGNER),
                "metric_evaluator_path": str(METRICS),
                "metric_evaluator_sha256": _sha256(METRICS),
            },
            "grid": {
                "noise_db": noise_values,
                "speech_start_padding": start_padding_values,
                "speech_end_padding": end_padding_values,
                "minimum_silence_seconds": arguments.minimum_silence,
                "selection_objective": "minimum mean DER on non-held-out calls",
                "tie_break": "lower numeric noise_db, then lower start padding, then lower end padding",
            },
            "configurations": configurations,
            "folds": folds,
            "macro_held_out_der": sum(held_out) / len(held_out),
            "claim_boundary": (
                "folds share one fixed-pause authored TTS generator and are not independent natural-call evidence"
            ),
        }
    finally:
        for path in temp.glob("*"):
            path.unlink()
        temp.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--transcriptions", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--noise-db", type=float, nargs="+", default=[-40, -45, -50, -55, -60])
    parser.add_argument("--speech-start-padding", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--speech-end-padding", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--minimum-silence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "vad-ranked-gap-loco-validation",
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
