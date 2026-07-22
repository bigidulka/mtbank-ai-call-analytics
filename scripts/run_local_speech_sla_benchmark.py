#!/usr/bin/env python3
"""Measures canonical local speech latency on an exact five-minute synthetic workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from mtbank_ai.runtime_secrets import SecretConfigurationError, require_environment_secret


class BenchmarkFailure(RuntimeError):
    pass


def _endpoint(base_url: str, *, bearer: bool = False) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("--base-url должен быть безопасным абсолютным HTTP(S) origin")
    if bearer and parsed.scheme != "https":
        raise ValueError("bearer --base-url должен использовать HTTPS")
    return f"{base_url.rstrip('/')}/v1/transcribe"


def _bearer_headers(api_key_env: str | None) -> dict[str, str] | None:
    if api_key_env is None:
        return None
    try:
        api_key = require_environment_secret(api_key_env, os.environ)
    except SecretConfigurationError as error:
        raise ValueError(str(error)) from error
    return {"Authorization": f"Bearer {api_key}"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _make_five_minutes(source: Path, destination: Path) -> None:
    command = (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stream_loop",
        "-1",
        "-i",
        str(source),
        "-t",
        "300",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(destination),
    )
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkFailure("не удалось создать five-minute synthetic workload") from error


def _duration_seconds(path: Path) -> float:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    )
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        duration = float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise BenchmarkFailure("не удалось определить duration workload") from error
    if not 299.9 <= duration <= 300.1:
        raise BenchmarkFailure("workload должен быть ровно 300 секунд")
    return duration


def benchmark(arguments: argparse.Namespace) -> tuple[int, dict[str, object]]:
    headers = _bearer_headers(getattr(arguments, "api_key_env", None))
    endpoint = _endpoint(arguments.base_url, bearer=headers is not None)
    with tempfile.TemporaryDirectory(prefix="mtbank-five-minute-") as directory:
        workload = Path(directory) / "five-minutes.wav"
        _make_five_minutes(arguments.audio, workload)
        duration = _duration_seconds(workload)
        started = time.monotonic()
        with workload.open("rb") as audio, httpx.Client(
            timeout=arguments.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = client.post(endpoint, files={"file": (workload.name, audio, "audio/wav")}, headers=headers)
        elapsed = time.monotonic() - started
        result = {
            "schema_version": 1,
            "kind": "canonical-five-minute-sla",
            "canonical_speech_path": True,
            "source_sha256": _sha256(arguments.audio),
            "workload_sha256": _sha256(workload),
            "audio_seconds": duration,
            "elapsed_seconds": round(elapsed, 3),
            "threshold_seconds": 60.0,
            "status_code": response.status_code,
            "within_sla": response.status_code == 200 and elapsed < 60.0,
        }
        return (0 if result["within_sla"] else 1), result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", help="имя переменной окружения bearer key для remote HTTPS")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    arguments = parser.parse_args()
    if not arguments.audio.is_file() or arguments.timeout_seconds <= 0:
        parser.error("--audio должен существовать, --timeout-seconds должен быть положительным")
    try:
        status, result = benchmark(arguments)
    except (BenchmarkFailure, ValueError) as error:
        status, result = 1, {
            "schema_version": 1,
            "kind": "canonical-five-minute-sla",
            "status": "failed",
            "reason": str(error),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed" if status == 0 else "failed", "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
