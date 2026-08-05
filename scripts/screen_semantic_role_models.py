#!/usr/bin/env python3
"""Screen gateway models for strict `submit_flat_turns` compliance before paired comparison.

The paired comparison in `compare_semantic_role_models.py` is fail-closed: one malformed
tool call aborts the whole study. Models differ in whether they honour a strict function
schema at all, and some honour it only intermittently, so screening that property first
keeps a long paired run from dying on a candidate that was never viable.

This measures schema compliance and latency only. It deliberately does not score role
quality — that requires the paired harness and held-out references.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, cast

from openai import OpenAI

if __package__:
    from .compare_semantic_role_models import TOOL_NAME, FlatDecision, _prompt, _tool, _validate_input
else:
    from compare_semantic_role_models import TOOL_NAME, FlatDecision, _prompt, _tool, _validate_input


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attempt(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    tool: Any,
    words: list[str],
) -> dict[str, object]:
    started = time.monotonic()
    try:
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
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": f"transport:{type(error).__name__}", "latency_ms": None}
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    calls = completion.choices[0].message.tool_calls or ()
    if len(calls) != 1 or cast(Any, calls[0]).function.name != TOOL_NAME:
        return {"ok": False, "reason": "no_single_tool_call", "latency_ms": latency_ms}
    try:
        decision = FlatDecision.model_validate_json(cast(Any, calls[0]).function.arguments, strict=True)
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": f"schema:{type(error).__name__}", "latency_ms": latency_ms}
    expected = 0
    for turn in decision.turns:
        if turn.start_word_index != expected:
            return {"ok": False, "reason": "coverage:gap_overlap_or_reorder", "latency_ms": latency_ms}
        expected = turn.end_word_index + 1
    if expected != len(words):
        return {"ok": False, "reason": "coverage:incomplete", "latency_ms": latency_ms}
    return {
        "ok": True,
        "reason": None,
        "latency_ms": latency_ms,
        "turns": len(decision.turns),
        "mean_confidence": statistics.fmean(turn.confidence for turn in decision.turns),
    }


def _screen_model(
    model: str,
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    prompt: str,
    tool: Any,
    files: tuple[dict[str, object], ...],
    attempts: int,
) -> dict[str, object]:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=0)
    results: list[dict[str, object]] = []
    try:
        for source in files:
            words = str(source["asr_text"]).split()
            for _ in range(attempts):
                outcome = _attempt(client, model=model, prompt=prompt, tool=tool, words=words)
                outcome["id"] = source["id"]
                results.append(outcome)
    finally:
        client.close()
    valid = [item for item in results if item["ok"]]
    latencies = [float(cast(float, item["latency_ms"])) for item in results if item["latency_ms"] is not None]
    return {
        "model": model,
        "attempts": len(results),
        "valid": len(valid),
        "compliance": len(valid) / len(results) if results else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "mean_turns": statistics.fmean(float(cast(float, item["turns"])) for item in valid) if valid else None,
        "mean_confidence": (
            statistics.fmean(float(cast(float, item["mean_confidence"])) for item in valid) if valid else None
        ),
        "failure_reasons": sorted({str(item["reason"]) for item in results if not item["ok"]}),
        "attempts_detail": results,
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    files = _validate_input(arguments.input)
    if arguments.files:
        wanted = {name.strip() for name in arguments.files.split(",") if name.strip()}
        files = tuple(source for source in files if str(source["id"]) in wanted)
        if not files:
            raise ValueError("no input file matched --files")
    models = tuple(dict.fromkeys(name.strip() for name in arguments.models.split(",") if name.strip()))
    if not models:
        raise ValueError("at least one model is required")
    api_key = os.environ.get(arguments.api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{arguments.api_key_env} is required")
    prompt, prompt_hash, tool = _prompt()
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.concurrency) as pool:
        futures = {
            pool.submit(
                _screen_model,
                model,
                base_url=arguments.base_url,
                api_key=api_key,
                timeout_seconds=arguments.timeout_seconds,
                prompt=prompt,
                tool=tool,
                files=files,
                attempts=arguments.attempts,
            ): model
            for model in models
        }
        screened = [future.result() for future in concurrent.futures.as_completed(futures)]
    screened.sort(key=lambda item: (-float(cast(float, item["compliance"])), str(item["model"])))
    return {
        "schema_version": 1,
        "kind": "semantic-role-model-schema-screen",
        "status": "completed",
        "scope": "strict tool-schema compliance and latency only; no role-quality claim",
        "provenance": {
            "input_path": str(arguments.input),
            "input_sha256": _sha256(arguments.input),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "prompt_hash": prompt_hash,
            "base_url_sha256": hashlib.sha256(arguments.base_url.encode()).hexdigest(),
        },
        "files": [str(source["id"]) for source in files],
        "attempts_per_file": arguments.attempts,
        "screened": screened,
        "claim_boundary": (
            "Compliance is measured on a small attempt budget, so a perfect score bounds but does not prove "
            "reliability. Latency is observational provider timing, not an SLA."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-url", default="https://llmass.arbitron.dev/v1")
    parser.add_argument("--api-key-env", default="MODEL_COMPARE_API_KEY")
    parser.add_argument("--models", required=True)
    parser.add_argument("--files", default=None, help="Comma-separated input IDs; default is every input file.")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.attempts < 1 or arguments.attempts > 10:
        parser.error("attempts must be in [1, 10]")
    if arguments.concurrency < 1 or arguments.concurrency > 8:
        parser.error("concurrency must be in [1, 8]")
    try:
        result = run(arguments)
        status = 0
    except Exception as error:  # noqa: BLE001
        result = {
            "schema_version": 1,
            "kind": "semantic-role-model-schema-screen",
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
