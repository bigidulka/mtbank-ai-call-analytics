#!/usr/bin/env python3
"""Evaluate pause-derived turns followed by text-only LLM role attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

if __package__:
    from .evaluate_semantic_role_attribution import (
        SemanticRoleDecision,
        _prompt_bundle,
        _tool_payload,
    )
    from .evaluate_speech import Segment, _counts_json, corpus_wer, speaker_attributed_wer, time_weighted_role_accuracy
else:
    from evaluate_semantic_role_attribution import SemanticRoleDecision, _prompt_bundle, _tool_payload
    from evaluate_speech import Segment, _counts_json, corpus_wer, speaker_attributed_wer, time_weighted_role_accuracy

from openai import OpenAI

from mtbank_ai.runtime_secrets import require_environment_secret
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


def _pause_segments(file: dict[str, object], threshold: float) -> tuple[dict[str, object], ...]:
    words = cast(list[dict[str, Any]], file["words"])
    groups: list[list[dict[str, Any]]] = []
    for word in words:
        if groups and float(word["start"]) - float(groups[-1][-1]["end"]) < threshold:
            groups[-1].append(word)
        else:
            groups.append([word])
    return tuple(
        {
            "id": f"pause-{index}",
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(str(word["word"]) for word in group),
        }
        for index, group in enumerate(groups)
    )


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    prompt, prompt_hash, tool = _prompt_bundle()
    corpus = json.loads(arguments.word_corpus.read_text())["files"]
    client = OpenAI(
        api_key=require_environment_secret(arguments.api_key_env, __import__("os").environ),
        base_url=arguments.base_url,
        timeout=arguments.timeout_seconds,
        max_retries=0,
    )
    files = []
    total_seconds = correct_seconds = total_words = attributed_errors = 0.0
    try:
        for file in corpus:
            identifier = str(file["id"])
            segments = _pause_segments(file, arguments.pause_threshold)
            completion = client.chat.completions.create(
                model=arguments.model,
                messages=(
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps({"segments": segments}, ensure_ascii=False, separators=(",", ":")),
                    },
                ),
                tools=(_tool_payload(tool),),  # type: ignore[arg-type]
                tool_choice={"type": "function", "function": {"name": "submit_semantic_roles"}},
                temperature=0,
                max_tokens=4_000,
            )
            call = completion.choices[0].message.tool_calls[0]  # type: ignore[index]
            decision = SemanticRoleDecision.model_validate_json(call.function.arguments, strict=True)  # type: ignore[union-attr]
            if tuple(x.segment_id for x in decision.assignments) != tuple(str(x["id"]) for x in segments):
                raise ValueError("incomplete segment assignment")
            roles = {x.segment_id: x.role for x in decision.assignments}
            hypothesis = tuple(
                Segment(
                    str(x["id"]), cast(float, x["start"]), cast(float, x["end"]), roles[str(x["id"])], str(x["text"])
                )
                for x in segments
            )
            reference = _reference(arguments.manifest, identifier)
            role_accuracy = time_weighted_role_accuracy(reference, hypothesis)
            attributed = speaker_attributed_wer(reference, hypothesis)
            wer = corpus_wer(reference, hypothesis)
            seconds = sum(x.end - x.start for x in reference)
            total_seconds += seconds
            correct_seconds += role_accuracy * seconds
            total_words += attributed.reference_words
            attributed_errors += attributed.errors
            files.append(
                {
                    "id": identifier,
                    "pause_segments": len(segments),
                    "metrics": {
                        "wer": _counts_json(wer),
                        "time_weighted_role_accuracy": role_accuracy,
                        "speaker_attributed_wer": _counts_json(attributed),
                    },
                }
            )
    finally:
        client.close()
    return {
        "schema_version": 1,
        "kind": "pause-semantic-role-evaluation",
        "status": "completed",
        "pause_threshold_seconds": arguments.pause_threshold,
        "prompt_hash": prompt_hash,
        "files": files,
        "micro": {
            "time_weighted_role_accuracy": correct_seconds / total_seconds,
            "speaker_attributed_wer": attributed_errors / total_words,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--word-corpus", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pause-threshold", type=float, default=1.3)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "pause-semantic-role-evaluation",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
