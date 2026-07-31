#!/usr/bin/env python3
"""Evaluate VAD-anchor + flat transcript semantic timestamp reconstruction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI
from pydantic import Field, model_validator

from mtbank_ai.agent_runtime import PromptRegistry
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema
from mtbank_ai.domain.base import StrictFrozenModel
from mtbank_ai.runtime_secrets import require_environment_secret
from mtbank_ai.speech.dataset import validate_manifest

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

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "src" / "mtbank_ai" / "agents"
TOOL_NAME = "submit_vad_aligned_turns"
MAX_WORDS = 1_200
MAX_ANCHORS = 256


class VadAlignedTurn(StrictFrozenModel):
    start_word_index: int = Field(ge=0, le=MAX_WORDS)
    end_word_index: int = Field(ge=0, le=MAX_WORDS)
    start_anchor_index: int = Field(ge=0, le=MAX_ANCHORS)
    end_anchor_index: int = Field(ge=0, le=MAX_ANCHORS)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def valid_ranges(self) -> VadAlignedTurn:
        if self.start_word_index > self.end_word_index or self.start_anchor_index > self.end_anchor_index:
            raise ValueError("turn range is reversed")
        return self


class VadAlignedDecision(StrictFrozenModel):
    turns: tuple[VadAlignedTurn, ...] = Field(min_length=1, max_length=MAX_ANCHORS)


def _prompt() -> tuple[str, str, FunctionToolSchema]:
    tool = FunctionToolSchema(
        name=TOOL_NAME,
        description="Submit contiguous word and VAD-anchor turn ranges with Operator/Client roles.",
        parameters=VadAlignedDecision.model_json_schema(),
    )
    bundle = PromptRegistry(PROMPT_ROOT).load(
        "vad_anchor_reconstructor",
        "v1",
        policy_inputs={
            "scope": "synthetic/no-PII",
            "roles": ["Оператор", "Клиент"],
            "speaker_labels": False,
            "word_timestamps": False,
            "vad_anchor_timestamps": True,
            "word_mutation": False,
        },
        tool_schemas=(tool,),
    )
    return bundle.text, bundle.reference.bundle_hash, tool


def _tool(tool: FunctionToolSchema) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        },
    }


def _reference(manifest: Path, identifier: str) -> tuple[Segment, ...]:
    entry = next(
        item
        for item in validate_manifest(manifest, require_release_corpus=True)
        if item.identifier == identifier and item.kind == "speech_reference"
    )
    raw = json.loads((entry.root / str(entry.raw["reference_path"])).read_text())["segments"]
    return tuple(Segment(str(x["id"]), x["start"], x["end"], x["speaker"], x["text"]) for x in raw)


def _coverage(decision: VadAlignedDecision, word_count: int, anchor_count: int) -> None:
    next_word = next_anchor = 0
    for turn in decision.turns:
        if turn.start_word_index != next_word or turn.start_anchor_index != next_anchor:
            raise ValueError("turn coverage has gap, overlap, or reorder")
        next_word = turn.end_word_index + 1
        next_anchor = turn.end_anchor_index + 1
    if next_word != word_count or next_anchor != anchor_count:
        raise ValueError("turns do not cover every word and anchor")


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    corpus = json.loads(arguments.corpus.read_text(encoding="utf-8"))
    prompt, prompt_hash, tool = _prompt()
    client = OpenAI(
        api_key=require_environment_secret(arguments.api_key_env, os.environ),
        base_url=arguments.base_url,
        timeout=arguments.timeout_seconds,
        max_retries=0,
    )
    files: list[dict[str, object]] = []
    miss = false_alarm = confusion = reference_seconds = correct_seconds = 0.0
    attributed_errors = attributed_reference_words = 0
    try:
        for file in corpus["files"]:
            words = str(file["text"]).split()
            anchors = cast(list[dict[str, object]], file["anchors"])
            if not words or len(words) > MAX_WORDS or not anchors or len(anchors) > MAX_ANCHORS:
                raise ValueError("input bounds exceeded")
            payload = {
                "words": [{"word_index": index, "word": word} for index, word in enumerate(words)],
                "anchors": anchors,
            }
            completion = client.chat.completions.create(
                model=arguments.model,
                messages=(
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
                ),
                tools=cast(Any, (_tool(tool),)),
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                temperature=0,
                max_tokens=8_000,
            )
            calls = completion.choices[0].message.tool_calls or ()
            call = cast(Any, calls[0]) if len(calls) == 1 else None
            if call is None or call.function.name != TOOL_NAME:
                raise ValueError("invalid terminal response")
            decision = VadAlignedDecision.model_validate_json(call.function.arguments, strict=True)
            _coverage(decision, len(words), len(anchors))
            hypothesis = tuple(
                Segment(
                    f"vad-{index}",
                    cast(float, anchors[turn.start_anchor_index]["start"]),
                    cast(float, anchors[turn.end_anchor_index]["end"]),
                    turn.role,
                    " ".join(words[turn.start_word_index : turn.end_word_index + 1]),
                )
                for index, turn in enumerate(decision.turns)
            )
            reference = _reference(arguments.manifest, str(file["id"]))
            wer = corpus_wer(reference, hypothesis)
            der = diarization_error_rate(reference, hypothesis)
            role_accuracy = time_weighted_role_accuracy(reference, hypothesis)
            attributed = speaker_attributed_wer(reference, hypothesis)
            miss += der["miss_seconds"]
            false_alarm += der["false_alarm_seconds"]
            confusion += der["confusion_seconds"]
            current_reference_seconds = der["reference_speaker_seconds"]
            reference_seconds += current_reference_seconds
            correct_seconds += role_accuracy * current_reference_seconds
            attributed_errors += attributed.errors
            attributed_reference_words += attributed.reference_words
            files.append(
                {
                    "id": file["id"],
                    "input_words": len(words),
                    "anchors": len(anchors),
                    "turns": len(decision.turns),
                    "metrics": {
                        "wer": _counts_json(wer),
                        "der": der,
                        "time_weighted_role_accuracy": role_accuracy,
                        "speaker_attributed_wer": _counts_json(attributed),
                    },
                }
            )
    finally:
        client.close()
    return {
        "schema_version": 1,
        "kind": "vad-anchor-role-evaluation",
        "status": "completed",
        "claim_boundary": "approximate timestamps from speaker-agnostic VAD anchors and semantic alignment",
        "prompt_hash": prompt_hash,
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
            "speaker_attributed_wer": attributed_errors / attributed_reference_words,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "vad-anchor-role-evaluation",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
