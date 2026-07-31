#!/usr/bin/env python3
"""Evaluate LLM semantic turn segmentation over ASR word timestamps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI
from pydantic import Field, model_validator

from mtbank_ai.agent_runtime import PromptRegistry
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema
from mtbank_ai.domain.base import StrictFrozenModel
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_environment_secret
from mtbank_ai.speech.dataset import validate_manifest

if __package__:
    from .evaluate_speech import (
        Segment,
        _counts_json,
        corpus_wer,
        speaker_attributed_wer,
        time_weighted_role_accuracy,
    )
else:
    from evaluate_speech import (
        Segment,
        _counts_json,
        corpus_wer,
        speaker_attributed_wer,
        time_weighted_role_accuracy,
    )

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "src" / "mtbank_ai" / "agents"
PROMPT_ID = "semantic_resegmenter"
PROMPT_VERSION = "v1"
TOOL_NAME = "submit_semantic_turns"
MAX_WORDS = 1_200
MAX_INPUT_CHARS = 70_000
MAX_OUTPUT_TOKENS = 8_000


class SemanticTurn(StrictFrozenModel):
    start_index: int = Field(ge=0, le=MAX_WORDS)
    end_index: int = Field(ge=0, le=MAX_WORDS)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def valid_range(self) -> SemanticTurn:
        if self.start_index > self.end_index:
            raise ValueError("turn index range is reversed")
        return self


class SemanticTurnDecision(StrictFrozenModel):
    turns: tuple[SemanticTurn, ...] = Field(min_length=1, max_length=MAX_WORDS)


class InputWord(StrictFrozenModel):
    index: int = Field(ge=0, le=MAX_WORDS)
    word: str = Field(min_length=1, max_length=200)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def valid_interval(self) -> InputWord:
        if self.start >= self.end:
            raise ValueError("word interval is reversed")
        return self


def _secret(name: str) -> str:
    try:
        return require_environment_secret(name, os.environ)
    except SecretConfigurationError as error:
        raise ValueError(f"{name} is unavailable") from error


def _prompt() -> tuple[str, str, FunctionToolSchema]:
    tool = FunctionToolSchema(
        name=TOOL_NAME,
        description="Submit contiguous semantic turns covering every input word exactly once.",
        parameters=SemanticTurnDecision.model_json_schema(),
    )
    bundle = PromptRegistry(PROMPT_ROOT).load(
        PROMPT_ID,
        PROMPT_VERSION,
        policy_inputs={
            "scope": "synthetic/no-PII",
            "roles": ["Оператор", "Клиент"],
            "max_words": MAX_WORDS,
            "speaker_labels_available": False,
            "acoustic_features_available": False,
            "word_mutation_allowed": False,
        },
        tool_schemas=(tool,),
    )
    return bundle.text, bundle.reference.bundle_hash, tool


def _tool_payload(tool: FunctionToolSchema) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": True,
        },
    }


def _load_corpus(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("word corpus requires non-empty files")
    return tuple(cast(dict[str, object], item) for item in files)


def _words(file: dict[str, object]) -> tuple[InputWord, ...]:
    raw_words = file.get("words")
    if not isinstance(raw_words, list) or not raw_words or len(raw_words) > MAX_WORDS:
        raise ValueError("word stream is invalid")
    words = tuple(
        InputWord.model_validate({"index": index, **cast(dict[str, object], raw)})
        for index, raw in enumerate(raw_words)
    )
    if any(current.start < previous.start for previous, current in zip(words, words[1:])):
        raise ValueError("word timestamps are not monotonic")
    return words


def _validate_coverage(decision: SemanticTurnDecision, count: int) -> None:
    expected = 0
    for turn in decision.turns:
        if turn.start_index != expected:
            raise ValueError("semantic turns have gap, overlap, or reorder")
        expected = turn.end_index + 1
    if expected != count:
        raise ValueError("semantic turns do not cover all words")


def _hypothesis(words: tuple[InputWord, ...], decision: SemanticTurnDecision) -> tuple[Segment, ...]:
    return tuple(
        Segment(
            identifier=f"semantic-{index}",
            start=words[turn.start_index].start,
            end=words[turn.end_index].end,
            speaker=turn.role,
            text=" ".join(word.word for word in words[turn.start_index : turn.end_index + 1]),
        )
        for index, turn in enumerate(decision.turns)
    )


def _reference(manifest: Path, identifier: str) -> tuple[Segment, ...]:
    entries = validate_manifest(manifest, require_release_corpus=True)
    entry = next(item for item in entries if item.identifier == identifier and item.kind == "speech_reference")
    path = entry.root / str(entry.raw["reference_path"])
    raw = json.loads(path.read_text(encoding="utf-8"))["segments"]
    return tuple(
        Segment(str(item["id"]), float(item["start"]), float(item["end"]), str(item["speaker"]), str(item["text"]))
        for item in raw
    )


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    prompt_text, bundle_hash, tool = _prompt()
    client = OpenAI(
        api_key=_secret(arguments.api_key_env),
        base_url=arguments.base_url,
        max_retries=0,
        timeout=arguments.timeout_seconds,
    )
    outputs: list[dict[str, object]] = []
    total_reference_seconds = weighted_correct_seconds = 0.0
    total_reference_words = total_attributed_errors = 0
    started = time.monotonic()
    try:
        for file in _load_corpus(arguments.word_corpus):
            identifier = cast(str, file["id"])
            words = _words(file)
            public_words = tuple(word.model_dump(exclude_none=True) for word in words)
            encoded = json.dumps({"words": public_words}, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) > MAX_INPUT_CHARS:
                raise ValueError("word stream exceeds input bound")
            request_started = time.monotonic()
            completion = client.chat.completions.create(
                model=arguments.model,
                messages=(
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": encoded},
                ),
                tools=cast(Any, (_tool_payload(tool),)),
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            message = completion.choices[0].message
            calls = message.tool_calls or ()
            call = cast(Any, calls[0]) if len(calls) == 1 else None
            if message.content or call is None or call.function.name != TOOL_NAME:
                raise ValueError(f"{identifier}: invalid terminal response")
            decision = SemanticTurnDecision.model_validate_json(call.function.arguments, strict=True)
            _validate_coverage(decision, len(words))
            hypothesis = _hypothesis(words, decision)
            reference = _reference(arguments.manifest, identifier)
            wer = corpus_wer(reference, hypothesis)
            attributed = speaker_attributed_wer(reference, hypothesis)
            role_accuracy = time_weighted_role_accuracy(reference, hypothesis)
            reference_seconds = sum(segment.end - segment.start for segment in reference)
            total_reference_seconds += reference_seconds
            weighted_correct_seconds += role_accuracy * reference_seconds
            total_reference_words += attributed.reference_words
            total_attributed_errors += attributed.errors
            outputs.append(
                {
                    "id": identifier,
                    "input_words": len(words),
                    "semantic_turns": len(decision.turns),
                    "latency_ms": round((time.monotonic() - request_started) * 1000, 3),
                    "metrics": {
                        "wer": _counts_json(wer),
                        "time_weighted_role_accuracy": role_accuracy,
                        "speaker_attributed_wer": _counts_json(attributed),
                    },
                    "hypothesis_sha256": hashlib.sha256(
                        json.dumps([segment.__dict__ for segment in hypothesis], ensure_ascii=False).encode()
                    ).hexdigest(),
                }
            )
    finally:
        client.close()
    return {
        "schema_version": 1,
        "kind": "semantic-resegmentation-evaluation",
        "status": "completed",
        "scope": "synthetic/no-PII saved faster-whisper word timestamps",
        "claim_boundary": (
            "word-stream semantic segmentation; no acoustic speaker identity; "
            "saved words originated from canonical artifacts"
        ),
        "model": arguments.model,
        "prompt": {"id": PROMPT_ID, "version": PROMPT_VERSION, "bundle_hash": bundle_hash},
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "files": outputs,
        "micro": {
            "time_weighted_role_accuracy": weighted_correct_seconds / total_reference_seconds,
            "speaker_attributed_wer": total_attributed_errors / total_reference_words,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--word-corpus", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "semantic-resegmentation-evaluation",
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
