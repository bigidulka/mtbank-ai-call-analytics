from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import SecretStr

from mtbank_ai.speech.streaming import (
    InternalSpeechWebSocketAdapter,
    InternalSpeechWebSocketSettings,
    RemoteSpeechWebSocketAdapter,
    RemoteSpeechWebSocketSettings,
    StreamingAdapterUnavailable,
    StreamingProtocolError,
    StreamingStart,
    StreamingUpdate,
    parse_streaming_start,
    validate_stream_frame,
)
from services.speech.settings import SpeechStreamingSettings
from services.speech.streaming import (
    PersistentOggOpusDecoder,
    RollingStreamingSession,
    StreamingRuntimeLimits,
    _OggLogicalStreamValidator,
)


class _FakeWebSocket:
    def __init__(self, responses: list[str | bytes]) -> None:
        self.responses = responses
        self.sent: list[str | bytes] = []
        self.closed = False

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        if not self.responses:
            raise AssertionError("unexpected internal WebSocket receive")
        return self.responses.pop(0)

    async def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.exited = False

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.exited = True


class _SpeechVad:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.closed = False

    def is_speech(self, pcm_s16le: bytes) -> bool:
        self.calls.append(pcm_s16le)
        return True

    def close(self) -> None:
        self.closed = True


class _Transcriber:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self._texts = iter(texts)
        self.calls: list[bytes] = []

    async def transcribe(self, pcm_s16le: bytes) -> str:
        self.calls.append(pcm_s16le)
        return next(self._texts)


def _settings() -> InternalSpeechWebSocketSettings:
    return InternalSpeechWebSocketSettings(
        base_url="http://speech:8010/prefix",
        stream_path="/v1/stream",
        open_timeout_seconds=1.0,
        ping_interval_seconds=1.0,
        ping_timeout_seconds=1.0,
        close_timeout_seconds=1.0,
        max_message_bytes=65_540,
    )


def _limits() -> StreamingRuntimeLimits:
    return StreamingRuntimeLimits(
        max_frame_bytes=64 * 1024,
        max_session_bytes=10 * 1024 * 1024,
        max_duration_seconds=300.0,
        processing_timeout_seconds=1.0,
        ffmpeg_timeout_seconds=1.0,
        max_decoder_output_bytes=10 * 1024 * 1024,
        max_decoder_stderr_bytes=64 * 1024,
    )


def _ogg_page(serial_number: int, page_sequence: int, header_type: int, payload: bytes) -> bytes:
    lacing = bytearray()
    remaining = len(payload)
    while remaining >= 255:
        lacing.append(255)
        remaining -= 255
    lacing.append(remaining)
    return (
        b"OggS"
        + b"\x00"
        + bytes((header_type,))
        + b"\x00" * 8
        + serial_number.to_bytes(4, "little")
        + page_sequence.to_bytes(4, "little")
        + b"\x00" * 4
        + bytes((len(lacing),))
        + bytes(lacing)
        + payload
    )


def test_internal_adapter_disables_proxy_and_closes_after_terminal_message() -> None:
    async def scenario() -> None:
        websocket = _FakeWebSocket(
            [
                json.dumps({"type": "started", "sequence": 0}),
                json.dumps(
                    {
                        "type": "update",
                        "sequence": 1,
                        "text": "частичный текст",
                        "stable_prefix": True,
                        "final": False,
                    }
                ),
                json.dumps({"type": "ack", "sequence": 1}),
                json.dumps(
                    {
                        "type": "update",
                        "sequence": 1,
                        "text": "итоговый текст",
                        "stable_prefix": True,
                        "final": True,
                    }
                ),
                json.dumps({"type": "finished", "sequence": 2}),
            ]
        )
        connection = _FakeConnection(websocket)
        captured: dict[str, Any] = {}

        def connector(*args: object, **kwargs: object) -> _FakeConnection:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return connection

        adapter = InternalSpeechWebSocketAdapter(_settings(), connector=connector)
        session = await adapter.open(StreamingStart("pcm_s16le", 16_000, 1))
        updates = await session.push(b"\x00\x00", sequence=1)
        final_updates = await session.finish()

        assert captured["args"] == ("ws://speech:8010/prefix/v1/stream",)
        assert captured["kwargs"] == {
            "compression": None,
            "proxy": None,
            "open_timeout": 1.0,
            "ping_interval": 1.0,
            "ping_timeout": 1.0,
            "close_timeout": 1.0,
            "max_size": 65_540,
            "max_queue": 1,
            "write_limit": 65_540,
        }
        assert json.loads(websocket.sent[0]) == {
            "type": "start",
            "sequence": 0,
            "codec": "pcm_s16le",
            "sample_rate_hz": 16_000,
            "channels": 1,
        }
        assert websocket.sent[1] == b"\x00\x00\x00\x01\x00\x00"
        assert json.loads(websocket.sent[2]) == {"type": "end", "sequence": 2}
        assert updates == (StreamingUpdate(sequence=1, text="частичный текст"),)
        assert final_updates == (StreamingUpdate(sequence=1, text="итоговый текст", final=True),)
        assert websocket.closed
        assert connection.exited

    asyncio.run(scenario())


def test_remote_adapter_uses_wss_one_bearer_header_and_no_proxy_or_compression() -> None:
    async def scenario() -> None:
        websocket = _FakeWebSocket([json.dumps({"type": "started", "sequence": 0})])
        connection = _FakeConnection(websocket)
        captured: dict[str, Any] = {}

        def connector(*args: object, **kwargs: object) -> _FakeConnection:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return connection

        settings = RemoteSpeechWebSocketSettings(
            base_url="https://speech.example.test/api",
            stream_path="/v1/stream",
            api_key=SecretStr("N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"),
            open_timeout_seconds=1.0,
            ping_interval_seconds=1.0,
            ping_timeout_seconds=1.0,
            close_timeout_seconds=1.0,
            max_message_bytes=65_540,
        )
        session = await RemoteSpeechWebSocketAdapter(settings, connector=connector).open(
            StreamingStart("pcm_s16le", 16_000, 1)
        )
        await session.close()

        assert captured["args"] == ("wss://speech.example.test/api/v1/stream",)
        assert captured["kwargs"]["additional_headers"] == [
            ("Authorization", "Bearer N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F")
        ]
        assert captured["kwargs"]["compression"] is None
        assert captured["kwargs"]["proxy"] is None

    asyncio.run(scenario())


def test_internal_adapter_closes_an_open_connection_after_start_rejection() -> None:
    async def scenario() -> None:
        websocket = _FakeWebSocket([json.dumps({"type": "rejected"})])
        connection = _FakeConnection(websocket)
        adapter = InternalSpeechWebSocketAdapter(_settings(), connector=lambda *args, **kwargs: connection)

        with pytest.raises(StreamingAdapterUnavailable, match="rejected start"):
            await adapter.open(StreamingStart("pcm_s16le", 16_000, 1))

        assert websocket.closed
        assert connection.exited

    asyncio.run(scenario())


def test_streaming_contract_rejects_ambiguous_opus_booleans_and_raw_packets() -> None:
    with pytest.raises(StreamingProtocolError):
        StreamingStart("opus", 48_000, 1)  # type: ignore[arg-type]
    with pytest.raises(StreamingProtocolError):
        StreamingUpdate(sequence=True, text="частичный текст")  # type: ignore[arg-type]
    with pytest.raises(StreamingProtocolError):
        parse_streaming_start(
            {"type": "start", "sequence": 0, "codec": "opus", "sample_rate_hz": 48_000, "channels": 1}
        )
    with pytest.raises(StreamingProtocolError):
        parse_streaming_start(
            {"type": "start", "sequence": False, "codec": "pcm_s16le", "sample_rate_hz": 16_000, "channels": True}
        )
    with pytest.raises(StreamingProtocolError, match="whole samples"):
        validate_stream_frame(StreamingStart("pcm_s16le", 16_000, 1), b"\x00", first_frame=True)
    with pytest.raises(StreamingProtocolError, match="Ogg page"):
        validate_stream_frame(StreamingStart("ogg_opus", 48_000, 1), b"raw-opus", first_frame=True)


def test_ogg_validator_allows_one_complete_logical_stream_only() -> None:
    first = _ogg_page(7, 0, 0x02, b"OpusHead" + b"\x00" * 11)
    last = _ogg_page(7, 1, 0x04, b"\x00")
    valid = _OggLogicalStreamValidator(64 * 1024)

    assert valid.feed(first[:10]) == b""
    assert valid.feed(first[10:]) == first
    assert valid.feed(last) == last
    valid.finish()

    chained = _OggLogicalStreamValidator(64 * 1024)
    chained.feed(first)
    with pytest.raises(StreamingAdapterUnavailable, match="discontinuous"):
        chained.feed(_ogg_page(8, 1, 0x04, b"\x00"))


def test_internal_streaming_settings_require_a_trusted_base_url_and_safe_path() -> None:
    with pytest.raises(ValueError, match="trusted HTTP"):
        InternalSpeechWebSocketSettings(
            base_url="http://speech.example.test:8010",
            stream_path="/v1/stream",
            open_timeout_seconds=1.0,
            ping_interval_seconds=1.0,
            ping_timeout_seconds=1.0,
            close_timeout_seconds=1.0,
            max_message_bytes=1,
        )
    with pytest.raises(ValueError, match="service path"):
        InternalSpeechWebSocketSettings(
            base_url="http://speech:8010",
            stream_path="//other-host/v1/stream",
            open_timeout_seconds=1.0,
            ping_interval_seconds=1.0,
            ping_timeout_seconds=1.0,
            close_timeout_seconds=1.0,
            max_message_bytes=1,
        )


def test_rolling_session_emits_first_partial_then_commits_common_prefix() -> None:
    async def scenario() -> None:
        transcriber = _Transcriber(("привет мир", "привет мир клиент"))
        session = RollingStreamingSession(
            start=StreamingStart("pcm_s16le", 16_000, 1),
            limits=_limits(),
            transcriber=transcriber,
            rolling_window_seconds=1.0,
            rolling_step_seconds=512 / 16_000,
            rolling_call_timeout_seconds=1.0,
            max_rolling_calls_per_session=3,
            max_rolling_audio_seconds_per_session=3.0,
            pcm_energy_threshold=0,
            decoder=None,
        )

        first = await session.push(b"\x01\x00" * 512, sequence=1)
        second = await session.push(b"\x01\x00" * 512, sequence=2)
        final = await session.finish()

        assert first == (StreamingUpdate(sequence=1, text="привет мир", stable_prefix=False),)
        assert second == (StreamingUpdate(sequence=2, text="привет мир", stable_prefix=True),)
        assert final == (StreamingUpdate(sequence=2, text="привет мир клиент", final=True),)
        assert len(transcriber.calls) == 2

    asyncio.run(scenario())


def test_rolling_session_calls_groq_for_short_pcm_on_finish() -> None:
    async def scenario() -> None:
        transcriber = _Transcriber(("короткий фрагмент",))
        session = RollingStreamingSession(
            start=StreamingStart("pcm_s16le", 16_000, 1),
            limits=_limits(),
            transcriber=transcriber,
            rolling_window_seconds=1.0,
            rolling_step_seconds=1.0,
            rolling_call_timeout_seconds=1.0,
            max_rolling_calls_per_session=1,
            max_rolling_audio_seconds_per_session=1.0,
            pcm_energy_threshold=0,
            decoder=None,
        )

        assert await session.push(b"\x01\x00", sequence=1) == ()
        assert await session.finish() == (
            StreamingUpdate(sequence=1, text="короткий фрагмент", stable_prefix=False, final=True),
        )
        assert transcriber.calls == [b"\x01\x00"]

    asyncio.run(scenario())


def test_streaming_settings_bound_decoded_pcm_and_update_text() -> None:
    settings = SpeechStreamingSettings()

    assert not settings.enabled
    assert settings.max_session_bytes == 10 * 1024 * 1024
    assert settings.max_session_bytes >= int(settings.max_duration_seconds * 16_000 * 2)
    assert settings.max_decoder_output_bytes >= 300 * 16_000 * 2
    with pytest.raises(ValueError, match="max_session_bytes"):
        SpeechStreamingSettings(max_session_bytes=5 * 1024 * 1024)
    with pytest.raises(ValueError, match="max_decoder_output_bytes"):
        SpeechStreamingSettings(max_decoder_output_bytes=1)
    with pytest.raises(ValueError, match="max_update_text_bytes"):
        SpeechStreamingSettings(max_frame_bytes=48 * 1024, max_update_text_bytes=64 * 1024)


def test_persistent_ogg_decoder_uses_direct_ffmpeg_argv_without_a_shell() -> None:
    argv = PersistentOggOpusDecoder(_limits()).argv

    assert argv[0] == "ffmpeg"
    assert "-nostdin" in argv
    assert "-xerror" in argv
    assert "pipe:0" in argv
    assert "pipe:1" in argv
    assert "shell" not in " ".join(argv)
