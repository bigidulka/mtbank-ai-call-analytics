#!/usr/bin/env python3
"""Запускает наблюдаемый NVIDIA WebSocket workload и экспортирует typed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from mtbank_ai.release.evidence import export_evidence, sha256, sha256_text, validate_evidence
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_environment_secret

ROOT = Path(__file__).parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_sha() -> str:
    completed = subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise ValueError("не удалось определить Git code SHA")
    return value


def _nvidia_observation() -> str:
    try:
        completed = subprocess.run(("nvidia-smi", "-L"), check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("GPU benchmark требует доступный NVIDIA driver и nvidia-smi") from error
    observation = completed.stdout.strip()
    if not observation.startswith("GPU "):
        raise RuntimeError("GPU benchmark не обнаружил NVIDIA GPU")
    return observation


def _image_digest(value: str) -> str:
    if not value.startswith("sha256:") or len(value) != 71 or any(char not in "0123456789abcdef" for char in value[7:]):
        raise ValueError("--image-digest должен быть фактическим sha256 image digest")
    return value


def _runtime_attestation(url: str, api_key_env: str, expected_digest: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/runtime"
    ):
        raise ValueError("--runtime-url должен быть безопасным HTTPS URL exact /v1/runtime")
    try:
        api_key = require_environment_secret(api_key_env, os.environ)
    except SecretConfigurationError as error:
        raise ValueError(str(error)) from error
    try:
        with httpx.Client(timeout=15.0, follow_redirects=False, trust_env=False) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise ValueError("remote runtime attestation is unavailable or invalid") from error
    if not isinstance(payload, dict) or set(payload) != {"runtime"} or not isinstance(payload["runtime"], dict):
        raise ValueError("remote runtime attestation has invalid schema")
    runtime = payload["runtime"]
    if runtime.get("image_digest") != expected_digest:
        raise ValueError("remote runtime attestation image digest does not match --image-digest")
    for component in ("asr", "diarization"):
        value = runtime.get(component)
        if not isinstance(value, dict) or set(value) != {"package", "package_version", "model_id", "model_revision"}:
            raise ValueError("remote runtime attestation component schema is invalid")
    return sha256(payload)


def _required_number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"WebSocket benchmark не вернул числовое поле {key}")
    return float(value)


def _required_positive_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"WebSocket benchmark не вернул положительное целое поле {key}")
    return value


def _benchmark(arguments: argparse.Namespace, output: Path) -> dict[str, object]:
    command = (
        sys.executable,
        str(ROOT / "scripts" / "run_websocket_benchmark.py"),
        "--url",
        arguments.websocket_url,
        "--origin",
        arguments.origin,
        "--audio",
        str(arguments.audio),
        "--api-key-env",
        arguments.api_key_env,
        "--frame-ms",
        str(arguments.frame_ms),
        "--output",
        str(output),
    )
    subprocess.run(command, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("WebSocket benchmark не вернул JSON object")
    if payload.get("diagnostic_only") is not True:
        raise ValueError("WebSocket benchmark не вернул diagnostic observations")
    for key in ("p50_ms", "p95_ms", "audio_seconds", "wall_latency_ms"):
        _required_number(payload, key)
    _required_positive_int(payload, "session_count")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--websocket-url", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MTBANK_API_KEY")
    parser.add_argument("--frame-ms", type=int, default=500)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument("--model-manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--workload-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.frame_ms <= 0 or not arguments.workload_revision.strip():
        parser.error("GPU workload parameters должны быть непустыми и положительными")
    if not arguments.audio.is_file() or not arguments.model_manifest.is_file():
        parser.error("--audio и --model-manifest должны существовать")

    nvidia = _nvidia_observation()
    image_digest = _image_digest(arguments.image_digest)
    runtime_attestation_sha256 = _runtime_attestation(arguments.runtime_url, arguments.api_key_env, image_digest)
    code_sha = _code_sha()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = arguments.output_dir / "websocket-diagnostic.json"
    benchmark = _benchmark(arguments, diagnostic_path)
    audio_seconds = _required_number(benchmark, "audio_seconds")
    wall_latency_ms = _required_number(benchmark, "wall_latency_ms")
    if audio_seconds <= 0 or wall_latency_ms <= 0:
        raise ValueError("WebSocket benchmark вернул неположительную duration/latency")
    provenance = {
        "image_digest": image_digest,
        "model_manifest_sha256": _sha256(arguments.model_manifest),
        "runner_id_sha256": sha256_text(socket.gethostname() + "\n" + nvidia),
        "runtime_attestation_sha256": runtime_attestation_sha256,
        "workload_revision": arguments.workload_revision,
    }
    gpu_evidence = export_evidence(
        kind="gpu-benchmark",
        code_sha=code_sha,
        provenance=provenance,
        metrics={
            "p50_ms": _required_number(benchmark, "p50_ms"),
            "p95_ms": _required_number(benchmark, "p95_ms"),
            "throughput_per_second": audio_seconds / (wall_latency_ms / 1000),
        },
        observations={},
    )
    websocket_evidence = export_evidence(
        kind="websocket-gpu-p95",
        code_sha=code_sha,
        provenance=provenance,
        metrics={
            "p95_ms": _required_number(benchmark, "p95_ms"),
            "session_count": _required_positive_int(benchmark, "session_count"),
        },
        observations={},
    )
    validate_evidence(gpu_evidence, expected_kind="gpu-benchmark", expected_code_sha=code_sha)
    validate_evidence(websocket_evidence, expected_kind="websocket-gpu-p95", expected_code_sha=code_sha)
    for name, evidence in (("gpu-benchmark.json", gpu_evidence), ("websocket-gpu-p95.json", websocket_evidence)):
        (arguments.output_dir / name).write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": "completed", "output_dir": str(arguments.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
