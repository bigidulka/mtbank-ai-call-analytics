#!/usr/bin/env python3
"""Paired comparison of Luna, Terra and Sol on identical flat ASR transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Literal, cast

from openai import OpenAI
from pydantic import Field, model_validator

from mtbank_ai.agent_runtime import PromptRegistry
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema
from mtbank_ai.domain.base import StrictFrozenModel

if __package__:
    from .evaluate_vad_rank_alignment import SemanticAssignment, _hypothesis
else:
    from evaluate_vad_rank_alignment import SemanticAssignment, _hypothesis

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "scope": "public external evaluation audio",
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


def _decision_once(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    tool: FunctionToolSchema,
    words: list[str],
) -> tuple[tuple[SemanticAssignment, ...], float]:
    started = time.monotonic()
    completion = client.chat.completions.create(
        model=model,
        messages=(
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"words": [{"index": index, "word": word} for index, word in enumerate(words)]},
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
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    calls = completion.choices[0].message.tool_calls or ()
    call = cast(Any, calls[0]) if len(calls) == 1 else None
    if call is None or call.function.name != TOOL_NAME:
        raise ValueError("invalid terminal response")
    parsed = FlatDecision.model_validate_json(call.function.arguments, strict=True)
    expected = 0
    assignments: list[SemanticAssignment] = []
    for turn in parsed.turns:
        if turn.start_word_index != expected:
            raise ValueError("turn coverage has gap, overlap, or reorder")
        expected = turn.end_word_index + 1
        assignments.append(
            SemanticAssignment(
                start_word_index=turn.start_word_index,
                end_word_index=turn.end_word_index,
                role=turn.role,
                confidence=turn.confidence,
            )
        )
    if expected != len(words):
        raise ValueError("turns do not cover every word")
    return tuple(assignments), latency_ms


def _decision(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    tool: FunctionToolSchema,
    words: list[str],
    max_attempts: int,
) -> tuple[tuple[SemanticAssignment, ...], float, int]:
    started = time.monotonic()
    for attempt in range(max_attempts):
        try:
            assignments, _ = _decision_once(
                client,
                model=model,
                prompt=prompt,
                tool=tool,
                words=words,
            )
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            retryable = status_code in {429, 500, 502, 503, 504} or type(error).__name__ in {
                "APIConnectionError",
                "APITimeoutError",
                "InternalServerError",
                "ValueError",
            }
            if not retryable or attempt + 1 >= max_attempts:
                raise
            time.sleep(min(8.0, 1.0 * (2**attempt)) + random.uniform(0.0, 0.25))
            continue
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        return assignments, elapsed_ms, attempt + 1
    raise RuntimeError("unreachable decision retry state")


def _signature(assignments: tuple[SemanticAssignment, ...]) -> str:
    payload = [
        [item.start_word_index, item.end_word_index, item.role, round(item.confidence, 6)] for item in assignments
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _role_signature(assignments: tuple[SemanticAssignment, ...]) -> str:
    payload = [[item.start_word_index, item.end_word_index, item.role] for item in assignments]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _word_roles(assignments: tuple[SemanticAssignment, ...], word_count: int) -> tuple[str, ...]:
    roles = [""] * word_count
    for assignment in assignments:
        for index in range(assignment.start_word_index, assignment.end_word_index + 1):
            roles[index] = assignment.role
    if any(not role for role in roles):
        raise ValueError("word-role coverage is incomplete")
    return tuple(roles)


def _fixed_alignment_status(
    *,
    text: str,
    duration: float,
    anchors: list[dict[str, object]],
    assignments: tuple[SemanticAssignment, ...],
) -> str:
    try:
        _hypothesis(
            {"text": text, "duration_seconds": duration, "anchors": anchors},
            assignments,
            speech_start_padding=0.2,
            speech_end_padding=0.9,
        )
    except ValueError as error:
        return f"failed:{error}"
    return "completed"


def _adaptive_alignment(
    *,
    text: str,
    duration: float,
    anchors: list[dict[str, object]],
    assignments: tuple[SemanticAssignment, ...],
) -> tuple[str, float | None, float | None]:
    for start, end in ((0.2, 0.9), (0.1, 0.2), (0.0, 0.0)):
        try:
            _hypothesis(
                {"text": text, "duration_seconds": duration, "anchors": anchors},
                assignments,
                speech_start_padding=start,
                speech_end_padding=end,
            )
        except ValueError:
            continue
        return "completed", start, end
    return "failed", None, None


def _validate_input(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ValueError("comparison input is incomplete")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("comparison input requires files")
    files: list[dict[str, object]] = []
    ids: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("comparison file must be object")
        identifier = raw.get("id")
        text = raw.get("asr_text")
        anchors = raw.get("vad_anchors")
        duration = raw.get("audio_seconds")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("comparison IDs are invalid")
        ids.add(identifier)
        if (
            not isinstance(text, str)
            or not text.strip()
            or hashlib.sha256(text.encode()).hexdigest() != raw.get("asr_text_sha256")
        ):
            raise ValueError(f"ASR text hash mismatch for {identifier}")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(f"VAD anchors missing for {identifier}")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration):
            raise ValueError(f"duration invalid for {identifier}")
        files.append(raw)
    return tuple(files)


def run(arguments: argparse.Namespace) -> dict[str, object]:
    files = _validate_input(arguments.input)
    models = tuple(arguments.models.split(","))
    if len(models) != 3 or len(set(models)) != 3:
        raise ValueError("exactly three unique models are required")
    if arguments.repeats < 2 or arguments.repeats > 5:
        raise ValueError("repeats must be in [2, 5]")
    if arguments.max_attempts < 1 or arguments.max_attempts > 3:
        raise ValueError("max_attempts must be in [1, 3]")
    api_key = os.environ.get(arguments.api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{arguments.api_key_env} is required")
    prompt, prompt_hash, tool = _prompt()
    client = OpenAI(api_key=api_key, base_url=arguments.base_url, timeout=arguments.timeout_seconds, max_retries=0)
    runs: list[dict[str, object]] = []
    schedule: list[tuple[int, str, int, dict[str, object]]] = [
        (repeat, model, source_index, source)
        for repeat in range(arguments.repeats)
        for source_index, source in enumerate(files)
        for model in models
    ]
    random.Random(arguments.schedule_seed).shuffle(schedule)
    try:
        for schedule_index, (repeat, model, source_index, source) in enumerate(schedule):
            identifier = str(source["id"])
            text = str(source["asr_text"])
            words = text.split()
            if not words or len(words) > MAX_WORDS:
                raise ValueError(f"word count invalid for {identifier}")
            assignments, latency_ms, attempts = _decision(
                client,
                model=model,
                prompt=prompt,
                tool=tool,
                words=words,
                max_attempts=arguments.max_attempts,
            )
            raw_duration = source["audio_seconds"]
            if not isinstance(raw_duration, (int, float)) or isinstance(raw_duration, bool):
                raise ValueError(f"duration invalid for {identifier}")
            duration = float(raw_duration)
            fixed_status = _fixed_alignment_status(
                text=text,
                duration=duration,
                anchors=cast(list[dict[str, object]], source["vad_anchors"]),
                assignments=assignments,
            )
            adaptive_status, adaptive_start, adaptive_end = _adaptive_alignment(
                text=text,
                duration=duration,
                anchors=cast(list[dict[str, object]], source["vad_anchors"]),
                assignments=assignments,
            )
            runs.append(
                {
                    "repeat": repeat,
                    "model": model,
                    "schedule_index": schedule_index,
                    "source_index": source_index,
                    "id": identifier,
                    "input_words": len(words),
                    "latency_ms": latency_ms,
                    "attempts": attempts,
                    "turns": len(assignments),
                    "mean_confidence": sum(item.confidence for item in assignments) / len(assignments),
                    "decision_sha256": _signature(assignments),
                    "role_boundary_sha256": _role_signature(assignments),
                    "assignments": [item.model_dump(mode="json") for item in assignments],
                    "word_roles": _word_roles(assignments, len(words)),
                    "fixed_alignment_status": fixed_status,
                    "adaptive_alignment_status": adaptive_status,
                    "adaptive_start_padding_seconds": adaptive_start,
                    "adaptive_end_padding_seconds": adaptive_end,
                }
            )
            if arguments.pacing_seconds > 0 and schedule_index + 1 < len(schedule):
                time.sleep(arguments.pacing_seconds)
    finally:
        client.close()
    paired: list[dict[str, object]] = []
    for repeat in range(arguments.repeats):
        for source in files:
            identifier = str(source["id"])
            relevant = [run for run in runs if run["repeat"] == repeat and run["id"] == identifier]
            if len(relevant) != len(models):
                raise ValueError("paired run coverage mismatch")
            disagreements: list[float] = []
            for left_index, left in enumerate(relevant):
                for right in relevant[left_index + 1 :]:
                    left_roles = cast(tuple[str, ...], left["word_roles"])
                    right_roles = cast(tuple[str, ...], right["word_roles"])
                    if len(left_roles) != len(right_roles):
                        raise ValueError("paired word-role lengths differ")
                    disagreements.append(
                        sum(a != b for a, b in zip(left_roles, right_roles, strict=True)) / len(left_roles)
                    )
            paired.append(
                {
                    "repeat": repeat,
                    "id": identifier,
                    "all_role_boundaries_identical": len({run["role_boundary_sha256"] for run in relevant}) == 1,
                    "mean_pairwise_word_role_disagreement": statistics.fmean(disagreements),
                    "fixed_alignment_outcomes": {str(run["model"]): run["fixed_alignment_status"] for run in relevant},
                }
            )

    def _run_float(run: dict[str, object], field: str) -> float:
        value = run[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"run {field} is not numeric")
        return float(value)

    def _run_int(run: dict[str, object], field: str) -> int:
        value = run[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"run {field} is not integer")
        return value

    model_summary: dict[str, object] = {}
    for model in models:
        relevant = [run for run in runs if run["model"] == model]
        model_summary[model] = {
            "runs": len(relevant),
            "request_attempts": sum(_run_int(run, "attempts") for run in relevant),
            "retried_runs": sum(_run_int(run, "attempts") > 1 for run in relevant),
            "median_latency_ms": statistics.median(_run_float(run, "latency_ms") for run in relevant),
            "mean_latency_ms": statistics.fmean(_run_float(run, "latency_ms") for run in relevant),
            "fixed_alignment_completed": sum(run["fixed_alignment_status"] == "completed" for run in relevant),
            "adaptive_alignment_completed": sum(run["adaptive_alignment_status"] == "completed" for run in relevant),
            "mean_turns": statistics.fmean(_run_int(run, "turns") for run in relevant),
            "mean_confidence": statistics.fmean(_run_float(run, "mean_confidence") for run in relevant),
        }
    within_model: dict[str, object] = {}
    for model in models:
        stable = 0
        for source in files:
            signatures = {
                run["role_boundary_sha256"] for run in runs if run["model"] == model and run["id"] == source["id"]
            }
            stable += int(len(signatures) == 1)
        within_model[model] = {"stable_files": stable, "files": len(files), "stability": stable / len(files)}
    for run in runs:
        run.pop("word_roles")
    return {
        "schema_version": 1,
        "kind": "paired-semantic-role-model-comparison",
        "status": "completed",
        "scope": "identical frozen external ASR text and VAD anchors; no references in inference",
        "provenance": {
            "input_path": str(arguments.input),
            "input_sha256": _sha256(arguments.input),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "prompt_hash": prompt_hash,
            "base_url_sha256": hashlib.sha256(arguments.base_url.encode()).hexdigest(),
            "schedule_seed": arguments.schedule_seed,
            "schedule_strategy": "seeded randomized model/file/repeat order",
        },
        "models": models,
        "repeats": arguments.repeats,
        "files": len(files),
        "runs": runs,
        "paired": paired,
        "model_summary": model_summary,
        "within_model": within_model,
        "claim_boundary": (
            "Randomized paired comparison measures association between selected model and outputs/latency for frozen "
            "inputs. Same-model nondeterminism remains, so causal model effects require comparing between-model and "
            "within-model variation. External reference quality is limited."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-url", default="https://llm.arbitron.dev/codex/v1")
    parser.add_argument("--api-key-env", default="MODEL_COMPARE_API_KEY")
    parser.add_argument("--models", default="gpt-5.6-luna,gpt-5.6-terra,gpt-5.6-sol")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--pacing-seconds", type=float, default=0.5)
    parser.add_argument("--schedule-seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.pacing_seconds) or arguments.pacing_seconds < 0:
        parser.error("pacing-seconds must be finite and non-negative")
    try:
        result = run(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "paired-semantic-role-model-comparison",
            "status": "failed",
            "reason": type(error).__name__,
            "message": str(error)[:300],
        }
        status = 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
