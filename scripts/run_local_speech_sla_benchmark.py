#!/usr/bin/env python3
"""Measures canonical local speech latency on an exact five-minute synthetic workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from mtbank_ai.public_endpoint import PublicEndpointError, require_public_dns_host
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_environment_secret
from mtbank_ai.speech.benchmark_workload import BenchmarkWorkloadError, duration_seconds, make_five_minutes

BenchmarkFailure = BenchmarkWorkloadError


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
    if bearer:
        try:
            require_public_dns_host(parsed.hostname or "", parsed.port or 443)
        except PublicEndpointError as error:
            raise ValueError(str(error)) from error
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


_make_five_minutes = make_five_minutes
_duration_seconds = duration_seconds


def benchmark(arguments: argparse.Namespace) -> tuple[int, dict[str, object]]:
    api_key_env = getattr(arguments, "api_key_env", None)
    endpoint = _endpoint(arguments.base_url, bearer=api_key_env is not None)
    headers = _bearer_headers(api_key_env)
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
