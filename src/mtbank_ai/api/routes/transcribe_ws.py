"""Authenticated, bounded WebSocket protocol for provisional local transcription."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, TypeVar, cast
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mtbank_ai.application.ports import AnalyzeCallPort, FileAnalyzeInput
from mtbank_ai.config import Settings, WebSocketSettings
from mtbank_ai.observability import Telemetry
from mtbank_ai.speech.streaming import (
    StreamingAdapterUnavailable,
    StreamingSpeechPort,
    StreamingSpeechSession,
    StreamingStart,
    StreamingUpdate,
    parse_streaming_binary_frame,
    parse_streaming_end,
    parse_streaming_start,
    validate_stream_frame,
)

router = APIRouter()

_CLOSE_UNAUTHENTICATED = 4401
_CLOSE_POLICY = 1008
_CLOSE_TOO_LARGE = 1009
_CLOSE_OVERLOADED = 1013
_CLEANUP_TIMEOUT_SECONDS = 0.05
_T = TypeVar("_T")


@dataclass(slots=True)
class _SessionState:
    sequence: int = 0
    bytes_received: int = 0
    codec: str = ""
    session: StreamingSpeechSession | None = None
    frames: list[bytes] | None = None


class WebSocketSessionManager:
    def __init__(self, max_sessions: int, telemetry: Telemetry) -> None:
        self._max_sessions = max_sessions
        self._active = 0
        self._lock = asyncio.Lock()
        self._telemetry = telemetry

    async def acquire(self) -> bool:
        async with self._lock:
            if self._active >= self._max_sessions:
                return False
            self._active += 1
            self._telemetry.metrics.gauge("mtbank_ws_active_sessions", self._active)
            return True

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
            self._telemetry.metrics.gauge("mtbank_ws_active_sessions", self._active)


@router.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket) -> None:
    settings = cast(Settings, websocket.app.state.settings)
    ws_settings = settings.websocket
    telemetry = cast(Telemetry, websocket.app.state.telemetry)
    manager = cast(WebSocketSessionManager, websocket.app.state.ws_sessions)
    streaming = cast(StreamingSpeechPort | None, websocket.app.state.streaming_speech)
    analyzer = cast(AnalyzeCallPort, websocket.app.state.analyzer)
    if not ws_settings.enabled or streaming is None:
        await websocket.close(code=_CLOSE_OVERLOADED)
        return
    if not _authorized(websocket, settings) or not _origin_allowed(websocket, ws_settings):
        await websocket.close(code=_CLOSE_UNAUTHENTICATED)
        return
    if not await manager.acquire():
        await websocket.close(code=_CLOSE_OVERLOADED)
        return

    state = _SessionState()
    request_id = uuid4()
    started = time.monotonic()
    deadline = asyncio.get_running_loop().time() + ws_settings.max_duration_seconds
    try:
        await _within_deadline(
            websocket.accept(headers=[(b"x-request-id", str(request_id).encode("ascii"))]),
            deadline,
        )
        with telemetry.context(request_id=request_id), telemetry.span("ws.transcribe"):
            start = await _receive_start(websocket, ws_settings, deadline)
            state.codec = start.codec
            state.frames = []
            state.session = await _within_deadline(streaming.open(start), deadline)
            await _send_json(websocket, {"type": "started", "sequence": state.sequence}, deadline)
            while True:
                message = await _within_deadline(websocket.receive(), deadline)
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("text") is not None:
                    action, sequence, frame = _parse_text_frame(cast(str, message["text"]), state.sequence + 1)
                    if action == "end":
                        await _finish(
                            websocket,
                            state,
                            analyzer,
                            request_id,
                            telemetry,
                            ws_settings.max_update_text_bytes,
                            deadline,
                        )
                        return
                elif message.get("bytes") is not None:
                    sequence, frame = _parse_binary_frame(cast(bytes, message["bytes"]), state.sequence + 1)
                else:
                    await websocket.close(code=_CLOSE_POLICY)
                    return
                if (
                    len(frame) > ws_settings.max_frame_bytes
                    or state.bytes_received + len(frame) > ws_settings.max_session_bytes
                ):
                    await websocket.close(code=_CLOSE_TOO_LARGE)
                    return
                validate_stream_frame(start, frame, first_frame=state.sequence == 0)
                state.sequence = sequence
                state.bytes_received += len(frame)
                assert state.frames is not None
                state.frames.append(frame)
                assert state.session is not None
                update_started = time.monotonic()
                updates = await _within_deadline(
                    state.session.push(frame, sequence=sequence),
                    min(deadline, asyncio.get_running_loop().time() + ws_settings.processing_timeout_seconds),
                )
                await _emit_updates(
                    websocket,
                    updates,
                    telemetry,
                    deadline,
                    max_update_text_bytes=ws_settings.max_update_text_bytes,
                    update_latency_seconds=max(0.0, time.monotonic() - update_started),
                )
                await _send_json(websocket, {"type": "ack", "sequence": sequence}, deadline)
    except TimeoutError:
        await _send_timeout(websocket)
    except WebSocketDisconnect:
        return
    except (ValueError, json.JSONDecodeError, binascii.Error):
        if websocket.client_state.name == "CONNECTED":
            await websocket.close(code=_CLOSE_POLICY)
    except RuntimeError:
        if websocket.client_state.name == "CONNECTED":
            await websocket.close(code=_CLOSE_OVERLOADED)
    finally:
        await _release_before_close(manager, state.session)
        telemetry.metrics.observe("mtbank_ws_session_latency_seconds", max(0.0, time.monotonic() - started))


async def _within_deadline(awaitable: Awaitable[_T], deadline: float) -> _T:
    async with asyncio.timeout_at(deadline):
        return await awaitable


async def _release_before_close(manager: WebSocketSessionManager, session: StreamingSpeechSession | None) -> None:
    release = asyncio.create_task(manager.release())
    try:
        await asyncio.shield(release)
    except asyncio.CancelledError:
        if session is not None:
            asyncio.create_task(_best_effort_close(session))
        raise
    if session is not None:
        await _best_effort_close(session)


async def _best_effort_close(session: StreamingSpeechSession) -> None:
    close = asyncio.create_task(session.close())
    try:
        async with asyncio.timeout(_CLEANUP_TIMEOUT_SECONDS):
            await asyncio.shield(close)
    except (TimeoutError, Exception):
        return


async def _send_timeout(websocket: WebSocket) -> None:
    if websocket.client_state.name != "CONNECTED":
        return
    try:
        async with asyncio.timeout(_CLEANUP_TIMEOUT_SECONDS):
            await websocket.send_json({"type": "timeout"})
    except (TimeoutError, RuntimeError, WebSocketDisconnect):
        return


async def _receive_start(websocket: WebSocket, settings: WebSocketSettings, deadline: float) -> StreamingStart:
    del settings
    message = await _within_deadline(websocket.receive_text(), deadline)
    return parse_streaming_start(_json_object(message))


def _parse_text_frame(value: str, expected_sequence: int) -> tuple[str, int, bytes]:
    payload = _json_object(value)
    action = payload.get("type")
    if action == "end":
        parse_streaming_end(payload, expected_sequence)
        return "end", expected_sequence, b""
    if (
        set(payload) != {"type", "sequence", "data"}
        or action != "audio"
        or payload.get("sequence") != expected_sequence
    ):
        raise ValueError("invalid audio")
    encoded = payload["data"]
    if not isinstance(encoded, str):
        raise ValueError("invalid audio data")
    return "audio", expected_sequence, base64.b64decode(encoded, validate=True)


def _parse_binary_frame(value: bytes, expected_sequence: int) -> tuple[int, bytes]:
    return parse_streaming_binary_frame(value, expected_sequence)


async def _emit_updates(
    websocket: WebSocket,
    updates: tuple[StreamingUpdate, ...],
    telemetry: Telemetry,
    deadline: float,
    *,
    max_update_text_bytes: int,
    update_latency_seconds: float | None = None,
) -> None:
    for update in updates:
        if len(update.text.encode("utf-8")) > max_update_text_bytes:
            raise StreamingAdapterUnavailable("streaming update exceeded bound")
        await _send_json(
            websocket,
            {
                "type": "provisional_final" if update.final else "partial",
                "sequence": update.sequence,
                "text": update.text,
            },
            deadline,
        )
        kind = "final" if update.final else "partial"
        telemetry.metrics.increment("mtbank_ws_updates_total", kind=kind)
        if update_latency_seconds is not None:
            telemetry.metrics.observe("mtbank_ws_update_latency_seconds", update_latency_seconds, kind=kind)


async def _send_json(websocket: WebSocket, payload: dict[str, object], deadline: float) -> None:
    await _within_deadline(websocket.send_json(payload), deadline)


async def _finish(
    websocket: WebSocket,
    state: _SessionState,
    analyzer: AnalyzeCallPort,
    request_id,
    telemetry: Telemetry,
    max_update_text_bytes: int,
    deadline: float,
) -> None:  # type: ignore[no-untyped-def]
    assert state.session is not None
    await _emit_updates(
        websocket,
        await _within_deadline(state.session.finish(), deadline),
        telemetry,
        deadline,
        max_update_text_bytes=max_update_text_bytes,
    )
    media_type = "audio/wav" if state.codec == "pcm_s16le" else "audio/ogg"
    content = b"".join(state.frames or ())
    if state.codec == "pcm_s16le":
        content = _pcm16_wave(content)
    result = await _within_deadline(
        analyzer.analyze(
            FileAnalyzeInput(
                filename=f"websocket.{'wav' if state.codec == 'pcm_s16le' else 'ogg'}",
                content_type=media_type,
                content=content,
            ),
            request_id=request_id,
        ),
        deadline,
    )
    await _send_json(
        websocket,
        {"type": "reconciled", "run_id": str(result.meta.run_id), "status": result.meta.status.value},
        deadline,
    )


def _pcm16_wave(payload: bytes) -> bytes:
    """Frames are raw 16 kHz mono PCM16; batch service receives a valid WAV container."""

    if len(payload) % 2:
        raise ValueError("PCM16 payload must have whole samples")
    byte_rate = 16_000 * 2
    return (
        b"RIFF"
        + (36 + len(payload)).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (16_000).to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + len(payload).to_bytes(4, "little")
        + payload
    )


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("message must be object")
    return parsed


def _authorized(websocket: WebSocket, settings: Settings) -> bool:
    import hmac

    value = websocket.headers.get("authorization", "")
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.casefold() != "bearer":
        return False
    try:
        return hmac.compare_digest(token.encode("ascii"), settings.api.api_key.get_secret_value().encode("ascii"))
    except UnicodeEncodeError:
        return False


def _origin_allowed(websocket: WebSocket, settings: WebSocketSettings) -> bool:
    origin = websocket.headers.get("origin")
    return origin is not None and origin in settings.allowed_origins
