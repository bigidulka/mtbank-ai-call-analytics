#!/usr/bin/env python3
"""Collect flat-text transcriptions from loopback ChatGPT Web gateway for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from mtbank_ai.speech.dataset import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("ChatGPT Web transcription collector accepts loopback HTTP only")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("base URL must not contain credentials, path, query, or fragment")
    return f"{base_url.rstrip('/')}/v1/audio/transcriptions"


def collect(arguments: argparse.Namespace) -> dict[str, object]:
    endpoint = _endpoint(arguments.base_url)
    entries = tuple(
        entry
        for entry in validate_manifest(arguments.manifest, require_release_corpus=True)
        if entry.kind == "speech_reference"
    )
    files: list[dict[str, object]] = []
    with httpx.Client(timeout=arguments.timeout_seconds, follow_redirects=False, trust_env=False) as client:
        for entry in entries:
            started = time.monotonic()
            with entry.path.open("rb") as audio:
                response = client.post(
                    endpoint,
                    files={"file": (entry.path.name, audio, "application/octet-stream")},
                    data={"model": arguments.model, "language": arguments.language, "response_format": "json"},
                )
            latency_ms = round((time.monotonic() - started) * 1000, 3)
            if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
                raise ValueError(f"{entry.identifier}: transcription request failed")
            payload = response.json()
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{entry.identifier}: transcription response has no text")
            files.append(
                {
                    "id": entry.identifier,
                    "audio_sha256": entry.raw["sha256"],
                    "duration_seconds": entry.raw["duration_seconds"],
                    "latency_ms": latency_ms,
                    "text": text.strip(),
                    "text_sha256": hashlib.sha256(text.strip().encode()).hexdigest(),
                }
            )
    return {
        "schema_version": 1,
        "kind": "chatgpt-web-flat-transcription-corpus",
        "status": "completed",
        "scope": "approved synthetic/no-PII",
        "provider_boundary": "unofficial ChatGPT Web endpoint through loopback gateway",
        "requested_model": arguments.model,
        "model_selection": "ignored by gateway; ChatGPT Web chooses internal model",
        "timestamp_support": False,
        "speaker_support": False,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "test_data/manifest.yaml")
    parser.add_argument("--base-url", default="http://127.0.0.1:37182")
    parser.add_argument("--model", default="gpt-4o-transcribe")
    parser.add_argument("--language", default="ru-RU")
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = collect(arguments)
        status = 0
    except Exception as error:
        result = {
            "schema_version": 1,
            "kind": "chatgpt-web-flat-transcription-corpus",
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
