#!/usr/bin/env python3
"""Считает noncanonical Groq-only WER без сохранения raw response."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from evaluate_speech import normalize_tokens, word_error_counts
from openai import AsyncOpenAI


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        key, _, value = raw.partition("=")
        values[key] = value
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return " ".join(segment["text"] for segment in payload["segments"])


async def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    environment = _load_env(arguments.env_file)
    client = AsyncOpenAI(
        api_key=environment["MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY"],
        base_url=environment["MTBANK_AGENT_RUNTIME__GATEWAY__BASE_URL"],
        max_retries=0,
    )
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["entries"] if entry["kind"] == "speech_reference"]
    results: list[dict[str, object]] = []
    aggregate = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    try:
        for entry in entries:
            audio_path = arguments.manifest.parent / entry["path"]
            reference_path = arguments.manifest.parent / entry["reference_path"]
            with audio_path.open("rb") as audio:
                response = await client.audio.transcriptions.create(
                    file=audio,
                    model=arguments.model,
                    language="ru",
                    response_format="json",
                    temperature=0.0,
                )
            hypothesis = response.text
            counts = word_error_counts(
                normalize_tokens(_reference_text(reference_path)),
                normalize_tokens(hypothesis),
            )
            for name in aggregate:
                aggregate[name] += getattr(counts, name)
            results.append(
                {
                    "id": entry["id"],
                    "audio_sha256": entry["sha256"],
                    "reference_sha256": entry["reference_sha256"],
                    "hypothesis_sha256": hashlib.sha256(hypothesis.encode("utf-8")).hexdigest(),
                    "duration_seconds": entry["duration_seconds"],
                    "wer": counts.rate,
                    "substitutions": counts.substitutions,
                    "deletions": counts.deletions,
                    "insertions": counts.insertions,
                    "reference_words": counts.reference_words,
                }
            )
    finally:
        await client.close()
    reference_words = aggregate["reference_words"]
    errors = aggregate["substitutions"] + aggregate["deletions"] + aggregate["insertions"]
    return {
        "schema_version": 1,
        "kind": "external-whisper-wer-baseline",
        "canonical_speech_path": False,
        "model": arguments.model,
        "language": "ru",
        "temperature": 0.0,
        "manifest_sha256": _sha256(arguments.manifest),
        "files": results,
        "micro": {**aggregate, "wer": errors / reference_words if reference_words else 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--manifest", type=Path, default=Path("test_data/manifest.yaml"))
    parser.add_argument("--model", default="whisper-large-v3")
    parser.add_argument("--output", type=Path, default=Path("test_data/evaluations/groq-whisper-large-v3.json"))
    arguments = parser.parse_args()
    result = asyncio.run(evaluate(arguments))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = result["files"]
    micro = result["micro"]
    if not isinstance(files, list) or not isinstance(micro, dict):
        raise RuntimeError("WER evaluation result имеет некорректную форму")
    print(
        json.dumps(
            {
                "model": result["model"],
                "files": len(files),
                "micro_wer": micro["wer"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
