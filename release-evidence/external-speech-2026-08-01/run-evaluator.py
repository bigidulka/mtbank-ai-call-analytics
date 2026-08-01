#!/usr/bin/env python3
"""Benchmark flat ChatGPT Web ASR + Luna roles + VAD on an external corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import httpx
from openai import OpenAI
from pydantic import Field, model_validator

from mtbank_ai.agent_runtime import PromptRegistry
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema
from mtbank_ai.domain.base import StrictFrozenModel

if __package__:
    from .evaluate_speech import ErrorCounts, _counts_json, normalize_tokens, word_error_counts
    from .evaluate_vad_rank_alignment import SemanticAssignment, _hypothesis
else:
    from evaluate_speech import ErrorCounts, _counts_json, normalize_tokens, word_error_counts
    from evaluate_vad_rank_alignment import SemanticAssignment, _hypothesis

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "src" / "mtbank_ai" / "agents"
TOOL_NAME = "submit_flat_turns"
MAX_WORDS = 1_200
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_EVENT = re.compile(r"silence_(start|end):\s*([0-9.]+)")


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


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("transcription gateway must be loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("transcription base URL is invalid")
    return f"{base_url.rstrip('/')}/v1/audio/transcriptions"


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


def _probe(path: Path) -> dict[str, object]:
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-show_entries",
            "format=duration,size,bit_rate,format_name",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if process.returncode != 0:
        raise ValueError(f"ffprobe failed for {path.name}")
    payload = json.loads(process.stdout)
    stream = payload["streams"][0]
    media = payload["format"]
    return {
        "duration_seconds": float(media["duration"]),
        "sample_rate_hz": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "codec": str(stream["codec_name"]),
        "container": str(media["format_name"]),
        "bytes": int(media["size"]),
        "bit_rate": int(media["bit_rate"]) if media.get("bit_rate") else None,
    }


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
        timeout=180,
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


def _validate_manifest(path: Path) -> tuple[dict[str, object], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("external manifest schema is invalid")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("external manifest requires files")
    files: list[dict[str, object]] = []
    ids: set[str] = set()
    root = path.parent.resolve()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ValueError("manifest file must be object")
        identifier = raw.get("id")
        relative = raw.get("path")
        license_name = raw.get("license")
        provenance = raw.get("provenance")
        if not all(
            isinstance(value, str) and value.strip() for value in (identifier, relative, license_name, provenance)
        ):
            raise ValueError("manifest identity/provenance fields are invalid")
        assert isinstance(identifier, str) and isinstance(relative, str)
        if identifier in ids:
            raise ValueError("manifest IDs must be unique")
        ids.add(identifier)
        audio = (root / relative).resolve()
        if root not in audio.parents or not audio.is_file() or audio.is_symlink():
            raise ValueError("manifest audio path is invalid")
        expected_hash = raw.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(audio) != expected_hash:
            raise ValueError(f"audio hash mismatch for {identifier}")
        reference = raw.get("reference_text")
        if reference is not None and (not isinstance(reference, str) or not reference.strip()):
            raise ValueError("reference_text must be non-empty or null")
        files.append({**raw, "id": identifier, "audio": audio, "reference_text": reference})
    return tuple(files)


def _transcribe(client: httpx.Client, endpoint: str, audio: Path, timeout_seconds: float) -> tuple[str, float]:
    started = time.monotonic()
    with audio.open("rb") as source:
        response = client.post(
            endpoint,
            files={"file": (audio.name, source, "application/octet-stream")},
            data={"model": "gpt-4o-transcribe", "language": "ru-RU", "response_format": "json"},
            timeout=timeout_seconds,
        )
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError(f"transcription failed for {audio.name}")
    payload = response.json()
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"transcription returned no text for {audio.name}")
    return text.strip(), latency_ms


def _roles(
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
        raise ValueError("Luna returned invalid terminal response")
    decision = FlatDecision.model_validate_json(call.function.arguments, strict=True)
    expected = 0
    assignments: list[SemanticAssignment] = []
    for turn in decision.turns:
        if turn.start_word_index != expected:
            raise ValueError("Luna turn coverage has gap, overlap, or reorder")
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
        raise ValueError("Luna turns do not cover every word")
    return tuple(assignments), latency_ms


def _sum_counts(current: ErrorCounts, addition: ErrorCounts) -> ErrorCounts:
    return ErrorCounts(
        substitutions=current.substitutions + addition.substitutions,
        deletions=current.deletions + addition.deletions,
        insertions=current.insertions + addition.insertions,
        reference_words=current.reference_words + addition.reference_words,
    )


def _align_with_bounded_padding(
    source: dict[str, object],
    assignments: tuple[SemanticAssignment, ...],
    *,
    start_padding: float,
    end_padding: float,
) -> tuple[tuple[Any, ...], float, float, bool]:
    candidates = (
        (start_padding, end_padding),
        (min(start_padding, 0.1), min(end_padding, 0.2)),
        (0.0, 0.0),
    )
    last_error: ValueError | None = None
    for current_start, current_end in dict.fromkeys(candidates):
        try:
            hypothesis = _hypothesis(
                source,
                assignments,
                speech_start_padding=current_start,
                speech_end_padding=current_end,
            )
            return hypothesis, current_start, current_end, (current_start, current_end) != candidates[0]
        except ValueError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def run(arguments: argparse.Namespace) -> dict[str, object]:
    files = _validate_manifest(arguments.manifest)
    endpoint = _endpoint(arguments.transcription_base_url)
    prompt, prompt_hash, tool = _prompt()
    api_key = os.environ.get(arguments.luna_api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{arguments.luna_api_key_env} is required")
    luna = OpenAI(
        api_key=api_key,
        base_url=arguments.luna_base_url,
        timeout=arguments.timeout_seconds,
        max_retries=0,
    )
    output_files: list[dict[str, object]] = []
    total_counts = ErrorCounts(0, 0, 0, 0)
    total_audio_seconds = total_asr_ms = total_luna_ms = total_vad_ms = 0.0
    run_started = time.monotonic()
    try:
        with httpx.Client(follow_redirects=False, trust_env=False) as transcriber:
            for item in files:
                identifier = str(item["id"])
                audio = cast(Path, item["audio"])
                media = _probe(audio)
                raw_duration = media["duration_seconds"]
                if not isinstance(raw_duration, (int, float)) or isinstance(raw_duration, bool):
                    raise ValueError(f"invalid probed duration for {identifier}")
                duration = float(raw_duration)
                total_audio_seconds += duration
                text, asr_ms = _transcribe(transcriber, endpoint, audio, arguments.timeout_seconds)
                words = text.split()
                if not words or len(words) > MAX_WORDS:
                    raise ValueError(f"flat transcript word count is invalid for {identifier}")
                assignments, luna_ms = _roles(luna, model=arguments.luna_model, prompt=prompt, tool=tool, words=words)
                vad_started = time.monotonic()
                anchors = _anchors(
                    audio,
                    duration,
                    noise_db=arguments.noise_db,
                    minimum_silence=arguments.minimum_silence,
                )
                source: dict[str, object] = {
                    "text": text,
                    "duration_seconds": duration,
                    "anchors": anchors,
                }
                alignment_status = "completed"
                alignment_reason: str | None = None
                padding_reduced = False
                applied_start_padding = arguments.speech_start_padding
                applied_end_padding = arguments.speech_end_padding
                try:
                    hypothesis, applied_start_padding, applied_end_padding, padding_reduced = (
                        _align_with_bounded_padding(
                            source,
                            assignments,
                            start_padding=arguments.speech_start_padding,
                            end_padding=arguments.speech_end_padding,
                        )
                    )
                except ValueError as error:
                    if not arguments.continue_on_alignment_failure:
                        raise
                    hypothesis = ()
                    alignment_status = "failed"
                    alignment_reason = str(error)[:200]
                vad_ms = round((time.monotonic() - vad_started) * 1000, 3)
                total_asr_ms += asr_ms
                total_luna_ms += luna_ms
                total_vad_ms += vad_ms
                reference = item.get("reference_text")
                counts = (
                    word_error_counts(normalize_tokens(reference), normalize_tokens(text))
                    if isinstance(reference, str)
                    else None
                )
                if counts is not None:
                    total_counts = _sum_counts(total_counts, counts)
                output_files.append(
                    {
                        "id": identifier,
                        "source": {
                            "url": item.get("source_url"),
                            "license": item["license"],
                            "provenance": item["provenance"],
                            "audio_sha256": item["sha256"],
                        },
                        "media": media,
                        "asr": {
                            "latency_ms": asr_ms,
                            "words": len(words),
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "wer": _counts_json(counts) if counts is not None else None,
                        },
                        "luna": {
                            "latency_ms": luna_ms,
                            "semantic_turns": len(assignments),
                            "mean_confidence": sum(item.confidence for item in assignments) / len(assignments),
                            "assignments": [item.model_dump(mode="json") for item in assignments],
                        },
                        "vad": {
                            "latency_ms": vad_ms,
                            "anchors": len(anchors),
                            "aligned_turns": len(hypothesis),
                            "status": alignment_status,
                            "reason": alignment_reason,
                            "applied_start_padding_seconds": applied_start_padding,
                            "applied_end_padding_seconds": applied_end_padding,
                            "padding_reduced_to_avoid_overlap": padding_reduced,
                        },
                        "hypothesis": [
                            {
                                "start": segment.start,
                                "end": segment.end,
                                "speaker": segment.speaker,
                                "text": segment.text,
                            }
                            for segment in hypothesis
                        ],
                    }
                )
    finally:
        luna.close()
    wall_ms = round((time.monotonic() - run_started) * 1000, 3)
    return {
        "schema_version": 1,
        "kind": "external-chatgpt-vad-luna-benchmark",
        "status": "completed",
        "scope": "publicly downloadable external evaluation audio; no private customer data",
        "provenance": {
            "manifest_path": str(arguments.manifest),
            "manifest_sha256": _sha256(arguments.manifest),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": _sha256(Path(__file__).resolve()),
            "prompt_hash": prompt_hash,
            "requested_asr_model": "gpt-4o-transcribe",
            "actual_asr_model": "unknown; unofficial ChatGPT Web gateway ignores model parameter",
            "luna_model": arguments.luna_model,
        },
        "configuration": {
            "vad_noise_db": arguments.noise_db,
            "vad_minimum_silence_seconds": arguments.minimum_silence,
            "speech_start_padding_seconds": arguments.speech_start_padding,
            "speech_end_padding_seconds": arguments.speech_end_padding,
        },
        "files": output_files,
        "aggregate": {
            "files": len(output_files),
            "audio_seconds": total_audio_seconds,
            "wall_latency_ms": wall_ms,
            "asr_latency_ms": total_asr_ms,
            "luna_latency_ms": total_luna_ms,
            "vad_alignment_latency_ms": total_vad_ms,
            "real_time_factor": wall_ms / 1000 / total_audio_seconds,
            "reference_text_files": sum(
                isinstance(file.get("asr"), dict) and cast(dict[str, object], file["asr"]).get("wer") is not None
                for file in output_files
            ),
            "micro_wer": _counts_json(total_counts) if total_counts.reference_words else None,
            "alignment_completed_files": sum(
                isinstance(file.get("vad"), dict) and cast(dict[str, object], file["vad"]).get("status") == "completed"
                for file in output_files
            ),
            "alignment_failed_files": sum(
                isinstance(file.get("vad"), dict) and cast(dict[str, object], file["vad"]).get("status") == "failed"
                for file in output_files
            ),
        },
        "claim_boundary": (
            "WER is available only where source reference text exists. External sources provide no trusted role/turn "
            "timestamps, so DER and role accuracy are not reported. Failed alignment is evidence against robustness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transcription-base-url", default="http://127.0.0.1:37182")
    parser.add_argument("--luna-base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--luna-api-key-env", default="CLIPROXY_LOCAL_API_KEY")
    parser.add_argument("--luna-model", default="gpt-5.6-luna")
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--noise-db", type=float, default=-45.0)
    parser.add_argument("--minimum-silence", type=float, default=0.25)
    parser.add_argument("--speech-start-padding", type=float, default=0.2)
    parser.add_argument("--speech-end-padding", type=float, default=0.9)
    parser.add_argument("--continue-on-alignment-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.timeout_seconds <= 0 or not all(
        math.isfinite(value) and value >= 0
        for value in (
            arguments.minimum_silence,
            arguments.speech_start_padding,
            arguments.speech_end_padding,
        )
    ):
        parser.error("timeouts and VAD/alignment bounds must be finite and non-negative")
    try:
        result = run(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "external-chatgpt-vad-luna-benchmark",
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
