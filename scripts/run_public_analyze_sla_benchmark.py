#!/usr/bin/env python3
"""Runs one privacy-safe exact-300-second public `/analyze` SLA observation."""

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

from mtbank_ai.domain.analysis import AnalyzeResponse
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_environment_secret
from scripts.run_local_speech_sla_benchmark import _duration_seconds, _make_five_minutes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _endpoint(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path not in {"", "/"}
    ):
        raise ValueError("--base-url должен быть безопасным HTTPS origin")
    return f"{value.rstrip('/')}/analyze"


def run(arguments: argparse.Namespace) -> tuple[int, dict[str, object]]:
    endpoint = _endpoint(arguments.base_url)
    try:
        api_key = require_environment_secret(arguments.api_key_env, os.environ)
    except SecretConfigurationError as error:
        raise ValueError(str(error)) from error
    with tempfile.TemporaryDirectory(prefix="mtbank-public-analyze-") as directory:
        workload = Path(directory) / "five-minutes.wav"
        _make_five_minutes(arguments.audio, workload)
        duration = _duration_seconds(workload)
        started = time.monotonic()
        with workload.open("rb") as audio, httpx.Client(
            timeout=arguments.timeout_seconds, follow_redirects=False, trust_env=False
        ) as client:
            response = client.post(
                endpoint,
                files={"file": (workload.name, audio, "audio/wav")},
                headers={"Authorization": f"Bearer {api_key}"},
            )
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        result: dict[str, object] = {
            "schema_version": 1,
            "kind": "public-analyze-five-minute-sla",
            "source_sha256": _sha256(arguments.audio),
            "workload_sha256": _sha256(workload),
            "audio_seconds": duration,
            "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
        }
        if response.status_code != 200:
            result["status"] = "failed"
            result["reason"] = "public_analyze_non_200"
            return 1, result
        try:
            analyzed = AnalyzeResponse.model_validate_json(response.content)
        except (TypeError, ValueError):
            result["status"] = "failed"
            result["reason"] = "public_analyze_invalid_schema"
            return 1, result
        versions = analyzed.meta.versions
        result.update(
            {
                "status": "completed",
                "component_metadata": {
                    "code_sha": versions.code_sha,
                    "asr": {"model_revision": versions.asr.model_revision},
                    "diarization": {"model_revision": versions.diarization.model_revision},
                },
            }
        )
        return 0, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    arguments = parser.parse_args()
    if not arguments.audio.is_file() or arguments.timeout_seconds <= 0:
        parser.error("--audio должен существовать, --timeout-seconds должен быть положительным")
    try:
        status, result = run(arguments)
    except ValueError as error:
        status, result = 1, {
            "schema_version": 1,
            "kind": "public-analyze-five-minute-sla",
            "status": "failed",
            "reason": str(error),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
