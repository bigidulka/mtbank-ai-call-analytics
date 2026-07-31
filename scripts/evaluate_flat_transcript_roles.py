#!/usr/bin/env python3
"""Evaluate LLM role reconstruction from flat text-only ASR transcripts."""

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
    from .evaluate_speech import Segment, _counts_json, speaker_attributed_wer
else:
    from evaluate_speech import Segment, _counts_json, speaker_attributed_wer

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "src" / "mtbank_ai" / "agents"
TOOL_NAME = "submit_flat_turns"
MAX_WORDS = 1_200


class FlatTurn(StrictFrozenModel):
    start_word_index: int = Field(ge=0, le=MAX_WORDS)
    end_word_index: int = Field(ge=0, le=MAX_WORDS)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def valid_range(self) -> FlatTurn:
        if self.start_word_index > self.end_word_index:
            raise ValueError("turn range is reversed")
        return self


class FlatDecision(StrictFrozenModel):
    turns: tuple[FlatTurn, ...] = Field(min_length=1, max_length=MAX_WORDS)


def _prompt() -> tuple[str, str, FunctionToolSchema]:
    tool = FunctionToolSchema(
        name=TOOL_NAME,
        description="Submit contiguous role turns covering every flat transcript word exactly once.",
        parameters=FlatDecision.model_json_schema(),
    )
    bundle = PromptRegistry(PROMPT_ROOT).load(
        "flat_transcript_reconstructor",
        "v1",
        policy_inputs={
            "scope": "synthetic/no-PII",
            "timestamps": False,
            "speaker_labels": False,
            "word_mutation": False,
            "roles": ["Оператор", "Клиент"],
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


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    corpus = json.loads(arguments.transcriptions.read_text())
    prompt, prompt_hash, tool = _prompt()
    client = OpenAI(
        api_key=require_environment_secret(arguments.api_key_env, os.environ),
        base_url=arguments.base_url,
        timeout=arguments.timeout_seconds,
        max_retries=0,
    )
    files = []
    total_errors = total_reference_words = 0
    try:
        for file in corpus["files"]:
            words = str(file["text"]).split()
            if not words or len(words) > MAX_WORDS:
                raise ValueError("flat transcript word count is invalid")
            completion = client.chat.completions.create(
                model=arguments.model,
                messages=(
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"words": [{"index": i, "word": word} for i, word in enumerate(words)]},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
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
            decision = FlatDecision.model_validate_json(call.function.arguments, strict=True)
            expected = 0
            for turn in decision.turns:
                if turn.start_word_index != expected:
                    raise ValueError("turn coverage has gap, overlap or reorder")
                expected = turn.end_word_index + 1
            if expected != len(words):
                raise ValueError("turns do not cover all words")
            hypothesis = tuple(
                Segment(
                    f"flat-{index}",
                    float(index),
                    float(index + 1),
                    turn.role,
                    " ".join(words[turn.start_word_index : turn.end_word_index + 1]),
                )
                for index, turn in enumerate(decision.turns)
            )
            reference = _reference(arguments.manifest, str(file["id"]))
            attributed = speaker_attributed_wer(reference, hypothesis)
            total_errors += attributed.errors
            total_reference_words += attributed.reference_words
            files.append(
                {
                    "id": file["id"],
                    "asr_latency_ms": file["latency_ms"],
                    "input_words": len(words),
                    "turns": len(decision.turns),
                    "speaker_attributed_wer": _counts_json(attributed),
                }
            )
    finally:
        client.close()
    return {
        "schema_version": 1,
        "kind": "flat-transcript-role-evaluation",
        "status": "completed",
        "claim_boundary": "no timestamps; role quality measured by speaker-attributed text only",
        "prompt_hash": prompt_hash,
        "files": files,
        "micro": {"speaker_attributed_wer": total_errors / total_reference_words},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--transcriptions", type=Path, required=True)
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
            "kind": "flat-transcript-role-evaluation",
            "status": "failed",
            "reason": type(error).__name__,
        }
        status = 1
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
