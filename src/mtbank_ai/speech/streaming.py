"""Bounded public-to-internal streaming speech contracts and WebSocket adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast
from urllib.parse import SplitResult, urlsplit, urlunsplit

from mtbank_ai.config import _is_internal_direct_ip, _is_internal_service_name, _validate_speech_service_path

StreamingCodec: TypeAlias = Literal["pcm_s16le", "ogg_opus"]


class StreamingProtocolError(ValueError):
    """A stream message or media contract violates the static protocol."""


class StreamingAdapterUnavailable(RuntimeError):
    """The internal streaming speech service is unavailable or rejected a session."""


@dataclass(frozen=True, slots=True)
class StreamingStart:
    codec: StreamingCodec
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        if (
            self.codec not in {"pcm_s16le", "ogg_opus"}
            or type(self.sample_rate_hz) is not int
            or type(self.channels) is not int
        ):
            raise StreamingProtocolError("unsupported streaming audio format")
        if self.codec == "pcm_s16le":
            valid = self.sample_rate_hz == 16_000 and self.channels == 1
        else:
            valid = self.sample_rate_hz == 48_000 and self.channels == 1
        if not valid:
            raise StreamingProtocolError("unsupported streaming audio format")


@dataclass(frozen=True, slots=True)
class StreamingUpdate:
    sequence: int
    text: str
    stable_prefix: bool = True
    final: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.sequence) is not int
            or not isinstance(self.text, str)
            or type(self.stable_prefix) is not bool
            or type(self.final) is not bool
            or self.sequence <= 0
            or not self.text.strip()
        ):
            raise StreamingProtocolError("streaming update must have sequence and text")


class StreamingSpeechSession(Protocol):
    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]: ...

    async def finish(self) -> tuple[StreamingUpdate, ...]: ...

    async def close(self) -> None: ...


class StreamingSpeechPort(Protocol):
    """A bounded rolling-ASR adapter emits provisional text before canonical reconciliation."""

    async def open(self, start: StreamingStart) -> StreamingSpeechSession: ...


class StreamingSpeechUnavailable:
    """Production default until local streaming artifacts and p95 are release-qualified."""

    async def open(self, start: StreamingStart) -> StreamingSpeechSession:
        del start
        raise StreamingAdapterUnavailable("streaming speech adapter is unavailable")


class _WebSocketConnection(Protocol):
    async def send(self, message: str | bytes) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


WebSocketConnector: TypeAlias = Callable[..., AbstractAsyncContextManager[_WebSocketConnection]]


@dataclass(frozen=True, slots=True)
class InternalSpeechWebSocketSettings:
    base_url: str
    stream_path: str
    open_timeout_seconds: float
    ping_interval_seconds: float
    ping_timeout_seconds: float
    close_timeout_seconds: float
    max_message_bytes: int
    max_queue: int = 1
    max_updates_per_operation: int = 4

    def __post_init__(self) -> None:
        if (
            self.open_timeout_seconds <= 0
            or self.ping_interval_seconds <= 0
            or self.ping_timeout_seconds <= 0
            or self.close_timeout_seconds <= 0
            or self.max_message_bytes <= 0
            or self.max_queue <= 0
            or self.max_updates_per_operation <= 0
        ):
            raise ValueError("internal streaming WebSocket limits must be positive")
        _trusted_internal_base_url(self.base_url)
        _validate_speech_service_path(self.stream_path)

    @property
    def url(self) -> str:
        parts = _trusted_internal_base_url(self.base_url)
        stream_path = _validate_speech_service_path(self.stream_path)
        base_path = parts.path.rstrip("/")
        return urlunsplit(("ws", parts.netloc, f"{base_path}{stream_path}", "", ""))


def _trusted_internal_base_url(value: str) -> SplitResult:
    try:
        parts = urlsplit(value)
        parts.port
    except ValueError:
        raise ValueError("internal streaming base URL is invalid") from None
    if (
        parts.scheme != "http"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or not (_is_internal_direct_ip(parts.hostname) or _is_internal_service_name(parts.hostname))
    ):
        raise ValueError("internal streaming requires a trusted HTTP base URL")
    if parts.path:
        _validate_speech_service_path(parts.path)
    return parts


class InternalSpeechWebSocketAdapter:
    """No-proxy client for the internal speech `/v1/stream` service only."""

    def __init__(
        self,
        settings: InternalSpeechWebSocketSettings,
        *,
        connector: WebSocketConnector | None = None,
    ) -> None:
        self._settings = settings
        self._connector = connector or _websockets_connector

    async def open(self, start: StreamingStart) -> StreamingSpeechSession:
        connection: AbstractAsyncContextManager[_WebSocketConnection] | None = None
        websocket: _WebSocketConnection | None = None
        try:
            connection = self._connector(
                self._settings.url,
                compression=None,
                proxy=None,
                open_timeout=self._settings.open_timeout_seconds,
                ping_interval=self._settings.ping_interval_seconds,
                ping_timeout=self._settings.ping_timeout_seconds,
                close_timeout=self._settings.close_timeout_seconds,
                max_size=self._settings.max_message_bytes,
                max_queue=self._settings.max_queue,
                write_limit=self._settings.max_message_bytes,
            )
            websocket = await connection.__aenter__()
            await _within_timeout(
                websocket.send(_json_message(_start_payload(start))),
                self._settings.open_timeout_seconds,
            )
            started = _json_object(await _within_timeout(websocket.recv(), self._settings.open_timeout_seconds))
            if started != {"type": "started", "sequence": 0}:
                raise StreamingAdapterUnavailable("internal streaming service rejected start")
            return _InternalSpeechWebSocketSession(websocket, connection, self._settings)
        except asyncio.CancelledError:
            await _abort_open_connection(websocket, connection, self._settings.close_timeout_seconds)
            raise
        except StreamingAdapterUnavailable:
            await _abort_open_connection(websocket, connection, self._settings.close_timeout_seconds)
            raise
        except Exception as error:
            await _abort_open_connection(websocket, connection, self._settings.close_timeout_seconds)
            raise StreamingAdapterUnavailable("internal streaming service is unavailable") from error


class _InternalSpeechWebSocketSession:
    def __init__(
        self,
        websocket: _WebSocketConnection,
        connection: AbstractAsyncContextManager[_WebSocketConnection],
        settings: InternalSpeechWebSocketSettings,
    ) -> None:
        self._websocket = websocket
        self._connection = connection
        self._settings = settings
        self._next_sequence = 1
        self._closed = False

    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]:
        if self._closed or sequence != self._next_sequence or not frame:
            raise StreamingAdapterUnavailable("internal streaming session rejected frame")
        try:
            await _within_timeout(
                self._websocket.send(sequence.to_bytes(4, "big") + frame),
                self._settings.ping_timeout_seconds,
            )
            updates = await self._receive_until("ack", sequence)
            self._next_sequence += 1
            return updates
        except StreamingAdapterUnavailable:
            await self.close()
            raise
        except Exception as error:
            await self.close()
            raise StreamingAdapterUnavailable("internal streaming frame failed") from error

    async def finish(self) -> tuple[StreamingUpdate, ...]:
        if self._closed:
            return ()
        sequence = self._next_sequence
        try:
            await _within_timeout(
                self._websocket.send(_json_message({"type": "end", "sequence": sequence})),
                self._settings.ping_timeout_seconds,
            )
            updates = await self._receive_until("finished", sequence)
            self._next_sequence += 1
            return updates
        except StreamingAdapterUnavailable:
            raise
        except Exception as error:
            raise StreamingAdapterUnavailable("internal streaming finish failed") from error
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await _within_timeout(self._websocket.close(), self._settings.close_timeout_seconds)
        except Exception:
            pass
        try:
            await self._connection.__aexit__(None, None, None)
        except Exception:
            return

    async def _receive_until(self, terminal_type: str, sequence: int) -> tuple[StreamingUpdate, ...]:
        updates: list[StreamingUpdate] = []
        while True:
            payload = _json_object(await _within_timeout(self._websocket.recv(), self._settings.ping_timeout_seconds))
            message_type = payload.get("type")
            if message_type == "update":
                if len(updates) >= self._settings.max_updates_per_operation:
                    raise StreamingAdapterUnavailable("internal streaming emitted too many updates")
                update = _update_from_payload(payload)
                if len(update.text.encode("utf-8")) > self._settings.max_message_bytes:
                    raise StreamingAdapterUnavailable("internal streaming update exceeded bound")
                updates.append(update)
                continue
            if payload == {"type": terminal_type, "sequence": sequence}:
                return tuple(updates)
            raise StreamingAdapterUnavailable("internal streaming protocol violation")


async def close_streaming_session(session: StreamingSpeechSession | None) -> None:
    if session is not None:
        await session.close()


async def _abort_open_connection(
    websocket: _WebSocketConnection | None,
    connection: AbstractAsyncContextManager[_WebSocketConnection] | None,
    timeout_seconds: float,
) -> None:
    if websocket is not None:
        try:
            await _within_timeout(websocket.close(), timeout_seconds)
        except Exception:
            pass
    if connection is not None:
        try:
            await _within_timeout(connection.__aexit__(None, None, None), timeout_seconds)
        except Exception:
            return


async def updates_from(session: StreamingSpeechSession, frame: bytes, sequence: int) -> AsyncIterator[StreamingUpdate]:
    for update in await session.push(frame, sequence=sequence):
        yield update


def parse_streaming_start(payload: Mapping[str, object]) -> StreamingStart:
    expected = {"type", "sequence", "codec", "sample_rate_hz", "channels"}
    sequence = payload.get("sequence")
    codec = payload.get("codec")
    sample_rate_hz = payload.get("sample_rate_hz")
    channels = payload.get("channels")
    if (
        set(payload) != expected
        or payload.get("type") != "start"
        or type(sequence) is not int
        or sequence != 0
        or not isinstance(codec, str)
        or codec not in {"pcm_s16le", "ogg_opus"}
        or type(sample_rate_hz) is not int
        or type(channels) is not int
    ):
        raise StreamingProtocolError("invalid streaming codec")
    return StreamingStart(
        codec=cast(StreamingCodec, codec),
        sample_rate_hz=sample_rate_hz,
        channels=channels,
    )


def parse_streaming_binary_frame(value: bytes, expected_sequence: int) -> tuple[int, bytes]:
    if len(value) <= 4:
        raise StreamingProtocolError("invalid streaming binary frame")
    sequence = int.from_bytes(value[:4], "big")
    if sequence != expected_sequence:
        raise StreamingProtocolError("unexpected streaming sequence")
    return sequence, value[4:]


def parse_streaming_end(payload: Mapping[str, object], expected_sequence: int) -> None:
    sequence = payload.get("sequence")
    if (
        set(payload) != {"type", "sequence"}
        or payload.get("type") != "end"
        or type(sequence) is not int
        or sequence != expected_sequence
    ):
        raise StreamingProtocolError("invalid streaming end")


def validate_stream_frame(start: StreamingStart, frame: bytes, *, first_frame: bool) -> None:
    if not frame:
        raise StreamingProtocolError("streaming audio frame is empty")
    if start.codec == "pcm_s16le":
        if len(frame) % 2:
            raise StreamingProtocolError("PCM16 frame must contain whole samples")
        return
    if first_frame and not frame.startswith(b"OggS"):
        raise StreamingProtocolError("ogg_opus stream must start with an Ogg page")


def _start_payload(start: StreamingStart) -> dict[str, object]:
    return {
        "type": "start",
        "sequence": 0,
        "codec": start.codec,
        "sample_rate_hz": start.sample_rate_hz,
        "channels": start.channels,
    }


def _update_from_payload(payload: Mapping[str, object]) -> StreamingUpdate:
    if set(payload) != {"type", "sequence", "text", "stable_prefix", "final"}:
        raise StreamingAdapterUnavailable("invalid internal streaming update")
    sequence = payload.get("sequence")
    text = payload.get("text")
    stable_prefix = payload.get("stable_prefix")
    final = payload.get("final")
    if (
        payload.get("type") != "update"
        or type(sequence) is not int
        or not isinstance(text, str)
        or type(stable_prefix) is not bool
        or type(final) is not bool
    ):
        raise StreamingAdapterUnavailable("invalid internal streaming update")
    try:
        return StreamingUpdate(sequence=sequence, text=text, stable_prefix=stable_prefix, final=final)
    except StreamingProtocolError as error:
        raise StreamingAdapterUnavailable("invalid internal streaming update") from error


def _json_message(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _json_object(value: str | bytes) -> dict[str, Any]:
    if not isinstance(value, str):
        raise StreamingAdapterUnavailable("internal streaming message must be text")
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise StreamingAdapterUnavailable("invalid internal streaming JSON") from error
    if not isinstance(parsed, dict):
        raise StreamingAdapterUnavailable("internal streaming JSON must be object")
    return parsed


async def _within_timeout(awaitable: Any, timeout_seconds: float) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except TimeoutError:
        raise StreamingAdapterUnavailable("internal streaming operation timed out") from None


def _websockets_connector(*args: Any, **kwargs: Any) -> AbstractAsyncContextManager[_WebSocketConnection]:
    try:
        from websockets.asyncio.client import connect  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise StreamingAdapterUnavailable("websockets dependency is unavailable") from error
    return connect(*args, **kwargs)
