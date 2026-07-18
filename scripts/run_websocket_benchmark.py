#!/usr/bin/env python3
"""Последовательный privacy-safe diagnostic client для `/ws/transcribe`."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import time
import wave
from pathlib import Path
from urllib.parse import urlsplit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


def _validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc or parsed.path != "/ws/transcribe":
        raise ValueError("--url должен быть абсолютным ws(s) URL exact `/ws/transcribe`")
    if parsed.query or parsed.fragment:
        raise ValueError("--url не должен содержать query или fragment")
    return value


def _read_pcm16(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16_000:
            raise ValueError("--audio должен быть mono 16 kHz PCM16 WAV")
        frames = source.readframes(source.getnframes())
    if not frames:
        raise ValueError("--audio не должен быть пустым")
    return frames, len(frames) // 2


def _json_object(raw: str) -> dict[str, object]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("WebSocket вернул некорректное сообщение")
    return payload


def _required_number(value: object, *, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} должен быть числом")
    return float(value)


def _safe_update(payload: dict[str, object], latency_ms: float) -> dict[str, object] | None:
    message_type = payload.get("type")
    if message_type not in {"partial", "provisional_final"}:
        return None
    text = payload.get("text")
    sequence = payload.get("sequence")
    if not isinstance(text, str) or not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ValueError("WebSocket вернул некорректный update")
    return {
        "sequence": sequence,
        "kind": message_type,
        "latency_ms": round(latency_ms, 3),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_bytes": len(text.encode("utf-8")),
    }


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    from websockets.asyncio.client import connect

    url = _validate_url(arguments.url)
    pcm, samples = _read_pcm16(arguments.audio)
    frame_bytes = arguments.frame_ms * 16_000 * 2 // 1000
    if frame_bytes <= 0 or arguments.frame_ms * 16_000 * 2 % 1000:
        raise ValueError("--frame-ms должен задавать целое число PCM samples")
    api_key = os.environ.get(arguments.api_key_env, "")
    if not api_key:
        raise ValueError(f"не задана переменная окружения {arguments.api_key_env}")
    headers = {"Authorization": f"Bearer {api_key}"}
    updates: list[dict[str, object]] = []
    session_started = time.monotonic()
    reconciliation_ms: float | None = None
    frame_count = 0
    async with connect(
        url,
        origin=arguments.origin,
        additional_headers=headers,
        max_size=arguments.max_message_bytes,
        proxy=None,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "sequence": 0,
                    "codec": "pcm_s16le",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                }
            )
        )
        if json.loads(await websocket.recv()) != {"type": "started", "sequence": 0}:
            raise ValueError("WebSocket не подтвердил start")
        for offset in range(0, len(pcm), frame_bytes):
            target = session_started + frame_count * arguments.frame_ms / 1000
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            frame_count += 1
            sent_at = time.monotonic()
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio",
                        "sequence": frame_count,
                        "data": base64.b64encode(pcm[offset : offset + frame_bytes]).decode("ascii"),
                    }
                )
            )
            while True:
                raw = await asyncio.wait_for(websocket.recv(), timeout=arguments.response_timeout_seconds)
                if not isinstance(raw, str):
                    raise ValueError("WebSocket вернул binary server frame")
                payload = _json_object(raw)
                update = _safe_update(payload, (time.monotonic() - sent_at) * 1000)
                if update is not None:
                    updates.append(update)
                if payload == {"type": "ack", "sequence": frame_count}:
                    break
        end_sent_at = time.monotonic()
        await websocket.send(json.dumps({"type": "end", "sequence": frame_count + 1}))
        while True:
            raw = await asyncio.wait_for(websocket.recv(), timeout=arguments.response_timeout_seconds)
            if not isinstance(raw, str):
                raise ValueError("WebSocket вернул binary server frame")
            payload = _json_object(raw)
            update = _safe_update(payload, (time.monotonic() - end_sent_at) * 1000)
            if update is not None:
                updates.append(update)
            if payload.get("type") == "reconciled":
                reconciliation_ms = round((time.monotonic() - end_sent_at) * 1000, 3)
                break
            if payload.get("type") == "timeout":
                raise TimeoutError("WebSocket session timed out")
    latencies = [_required_number(update.get("latency_ms"), field="latency_ms") for update in updates]
    return {
        "schema_version": 1,
        "kind": "websocket-streaming-diagnostic",
        "diagnostic_only": True,
        "evidence_eligible_for_gpu_gate": False,
        "route_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        "audio_sha256": _sha256(arguments.audio),
        "audio_seconds": samples / 16_000,
        "frame_seconds": arguments.frame_ms / 1000,
        "frames_sent": frame_count,
        "updates": updates,
        "p50_ms": _percentile(latencies, 0.5),
        "p95_ms": _percentile(latencies, 0.95),
        "max_ms": max(latencies, default=0.0),
        "reconciliation_ms": reconciliation_ms,
        "session_count": 1,
        "wall_latency_ms": round((time.monotonic() - session_started) * 1000, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MTBANK_API_KEY")
    parser.add_argument("--frame-ms", type=int, default=500)
    parser.add_argument("--response-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-message-bytes", type=int, default=98_304)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.frame_ms <= 0 or arguments.response_timeout_seconds <= 0 or arguments.max_message_bytes <= 0:
        parser.error("лимиты benchmark должны быть положительными")
    result = asyncio.run(run(arguments))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(arguments.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
