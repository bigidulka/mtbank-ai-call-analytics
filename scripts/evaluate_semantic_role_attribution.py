#!/usr/bin/env python3
"""Evaluate text-only LLM role attribution on approved synthetic/no-PII references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI
from pydantic import Field

from mtbank_ai.agent_runtime import PromptRegistry
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema
from mtbank_ai.domain.base import StrictFrozenModel
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_environment_secret
from mtbank_ai.speech.dataset import ManifestEntry, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "src" / "mtbank_ai" / "agents"
PROMPT_ID = "semantic_role_attributor"
PROMPT_VERSION = "v1"
TOOL_NAME = "submit_semantic_roles"
MAX_SEGMENTS = 128
MAX_INPUT_CHARS = 24_000
MAX_OUTPUT_TOKENS = 4_000


class SemanticAssignment(StrictFrozenModel):
    segment_id: str = Field(min_length=1, max_length=256)
    role: Literal["Оператор", "Клиент"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, max_length=400)


class SemanticRoleDecision(StrictFrozenModel):
    assignments: tuple[SemanticAssignment, ...] = Field(max_length=MAX_SEGMENTS)


def _required_secret(name: str) -> str:
    try:
        return require_environment_secret(name, os.environ)
    except SecretConfigurationError as error:
        raise ValueError(f"{name} is unavailable") from error


def _reference_entries(manifest: Path) -> tuple[ManifestEntry, ...]:
    return tuple(
        entry for entry in validate_manifest(manifest, require_release_corpus=True) if entry.kind == "speech_reference"
    )


def _prompt_bundle() -> tuple[str, str, FunctionToolSchema]:
    tool = FunctionToolSchema(
        name=TOOL_NAME,
        description="Submit one semantic Operator/Client assignment for every input segment.",
        parameters=SemanticRoleDecision.model_json_schema(),
    )
    bundle = PromptRegistry(PROMPT_ROOT).load(
        PROMPT_ID,
        PROMPT_VERSION,
        policy_inputs={
            "input_scope": "synthetic/no-PII",
            "max_segments": MAX_SEGMENTS,
            "max_input_chars": MAX_INPUT_CHARS,
            "roles": ["Оператор", "Клиент"],
            "speaker_labels_available": False,
            "acoustic_features_available": False,
            "turn_boundaries": "oracle_reference_segments",
        },
        tool_schemas=(tool,),
    )
    return bundle.text, bundle.reference.bundle_hash, tool


def _payload(reference: dict[str, object]) -> tuple[str, tuple[dict[str, object], ...]]:
    raw_segments = reference.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments or len(raw_segments) > MAX_SEGMENTS:
        raise ValueError("reference segments are invalid")
    segments: list[dict[str, object]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("reference segment is invalid")
        segments.append(
            {
                "id": raw["id"],
                "start": raw["start"],
                "end": raw["end"],
                "text": raw["text"],
            }
        )
    encoded = json.dumps({"segments": segments}, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > MAX_INPUT_CHARS:
        raise ValueError("reference conversation exceeds text-only input bound")
    return encoded, tuple(segments)


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


def _score(
    reference: dict[str, object],
    decision: SemanticRoleDecision,
    input_segments: tuple[dict[str, object], ...],
) -> dict[str, object]:
    reference_segments = reference["segments"]
    assert isinstance(reference_segments, list)
    expected_ids = tuple(str(segment["id"]) for segment in input_segments)
    assignments = decision.assignments
    actual_ids = tuple(item.segment_id for item in assignments)
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ValueError("model did not return every segment exactly once in input order")
    reference_by_id = {str(item["id"]): item for item in reference_segments if isinstance(item, dict)}
    correct_segments = 0
    correct_seconds = 0.0
    total_seconds = 0.0
    confidences: list[float] = []
    mistakes: list[dict[str, object]] = []
    for assignment in assignments:
        expected = reference_by_id[assignment.segment_id]
        duration = float(expected["end"]) - float(expected["start"])
        total_seconds += duration
        confidences.append(assignment.confidence)
        if assignment.role == expected["speaker"]:
            correct_segments += 1
            correct_seconds += duration
        else:
            mistakes.append(
                {
                    "segment_id": assignment.segment_id,
                    "expected": expected["speaker"],
                    "predicted": assignment.role,
                    "confidence": assignment.confidence,
                    "text_sha256": hashlib.sha256(str(expected["text"]).encode()).hexdigest(),
                }
            )
    return {
        "segments": len(assignments),
        "segment_accuracy": correct_segments / len(assignments),
        "time_weighted_role_accuracy": correct_seconds / total_seconds,
        "mean_confidence": sum(confidences) / len(confidences),
        "mistakes": mistakes,
    }


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    prompt_text, bundle_hash, tool = _prompt_bundle()
    client = OpenAI(
        api_key=_required_secret(arguments.api_key_env),
        base_url=arguments.base_url,
        max_retries=0,
        timeout=arguments.timeout_seconds,
    )
    files: list[dict[str, object]] = []
    total_segments = correct_segments = 0
    total_seconds = correct_seconds = 0.0
    started = time.monotonic()
    try:
        for entry in _reference_entries(arguments.manifest):
            reference_path = entry.root / str(entry.raw["reference_path"])
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            payload, input_segments = _payload(reference)
            completion = client.chat.completions.create(
                model=arguments.model,
                messages=(
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": payload},
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
                raise ValueError(f"{entry.identifier}: invalid terminal response")
            decision = SemanticRoleDecision.model_validate_json(call.function.arguments, strict=True)
            metrics = _score(reference, decision, input_segments)
            files.append({"id": entry.identifier, "metrics": metrics})
            segment_count = cast(int, metrics["segments"])
            segment_accuracy = cast(float, metrics["segment_accuracy"])
            total_segments += segment_count
            correct_segments += round(segment_accuracy * segment_count)
            for assignment in decision.assignments:
                expected = next(item for item in reference["segments"] if item["id"] == assignment.segment_id)
                duration = float(expected["end"]) - float(expected["start"])
                total_seconds += duration
                if assignment.role == expected["speaker"]:
                    correct_seconds += duration
    finally:
        client.close()
    return {
        "schema_version": 1,
        "kind": "semantic-role-attribution-evaluation",
        "status": "completed",
        "scope": "synthetic/no-PII",
        "condition": "oracle text segments; speaker labels and acoustic features withheld",
        "claim_boundary": "upper bound for semantic role attribution; not acoustic diarization",
        "model": arguments.model,
        "prompt": {"id": PROMPT_ID, "version": PROMPT_VERSION, "bundle_hash": bundle_hash},
        "latency_ms": round((time.monotonic() - started) * 1000, 3),
        "files": files,
        "micro": {
            "segments": total_segments,
            "segment_accuracy": correct_segments / total_segments,
            "time_weighted_role_accuracy": correct_seconds / total_seconds,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "semantic-role-attribution-evaluation",
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
