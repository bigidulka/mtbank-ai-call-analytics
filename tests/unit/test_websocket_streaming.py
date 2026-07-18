from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import cast
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import SecretStr

from mtbank_ai.api.main import create_app
from mtbank_ai.application.ports import AnalyzeCallPort, FileAnalyzeInput
from mtbank_ai.config import ApiSettings, DatabaseSettings, Settings, WebSocketSettings
from mtbank_ai.speech.streaming import StreamingStart, StreamingUpdate

_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


class _Ready:
    async def ping(self) -> bool:
        return True


class _StreamingSession:
    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]:
        assert frame == b"pcm0"
        return (StreamingUpdate(sequence=sequence, text="частичный текст", stable_prefix=False),)

    async def finish(self) -> tuple[StreamingUpdate, ...]:
        return (StreamingUpdate(sequence=2, text="итог", final=True),)

    async def close(self) -> None:
        return None


class _StreamingPort:
    async def open(self, start: StreamingStart) -> _StreamingSession:
        assert start.codec == "pcm_s16le"
        return _StreamingSession()


class _BlockingCloseSession(_StreamingSession):
    async def close(self) -> None:
        await asyncio.Event().wait()


class _BlockingStreamingPort:
    async def open(self, start: StreamingStart) -> _BlockingCloseSession:
        assert start.codec == "pcm_s16le"
        return _BlockingCloseSession()


class _BlockingAnalyzer:
    async def analyze(self, source: FileAnalyzeInput, *, request_id: UUID) -> SimpleNamespace:
        del source, request_id
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _Analyzer:
    def __init__(self) -> None:
        self.sources: list[FileAnalyzeInput] = []

    async def analyze(self, source: FileAnalyzeInput, *, request_id: UUID) -> SimpleNamespace:
        del request_id
        self.sources.append(source)
        return SimpleNamespace(meta=SimpleNamespace(run_id=UUID(int=1), status=SimpleNamespace(value="completed")))


def test_websocket_emits_partial_before_end_and_reconciles_canonical_batch() -> None:
    analyzer = _Analyzer()
    settings = Settings(
        environment="test",
        api=ApiSettings(api_key=SecretStr(_KEY)),
        database=DatabaseSettings(password=SecretStr("opaque-database-password")),
        websocket=WebSocketSettings(enabled=True, allowed_origins=("https://test.example",)),
    )
    app = create_app(
        settings=settings,
        analyzer=cast(AnalyzeCallPort, analyzer),
        readiness=_Ready(),
        streaming_speech=_StreamingPort(),
    )

    with TestClient(app).websocket_connect(
        "/ws/transcribe",
        headers={"Authorization": f"Bearer {_KEY}", "Origin": "https://test.example"},
    ) as websocket:
        websocket.send_json(
            {"type": "start", "sequence": 0, "codec": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1}
        )
        assert websocket.receive_json()["type"] == "started"
        websocket.send_json({"type": "audio", "sequence": 1, "data": base64.b64encode(b"pcm0").decode("ascii")})
        assert websocket.receive_json()["type"] == "partial"
        websocket.send_json({"type": "end", "sequence": 2})
        assert websocket.receive_json()["type"] == "provisional_final"
        assert websocket.receive_json()["type"] == "reconciled"

    assert analyzer.sources[0].content.endswith(b"pcm0")
    assert analyzer.sources[0].content.startswith(b"RIFF")


def _timeout_settings() -> Settings:
    return Settings(
        environment="test",
        api=ApiSettings(api_key=SecretStr(_KEY)),
        database=DatabaseSettings(password=SecretStr("opaque-database-password")),
        websocket=WebSocketSettings(
            enabled=True,
            allowed_origins=("https://test.example",),
            max_duration_seconds=0.02,
            max_sessions=1,
        ),
    )


def test_silent_websocket_deadline_releases_session_slot() -> None:
    app = create_app(
        settings=_timeout_settings(),
        analyzer=cast(AnalyzeCallPort, _Analyzer()),
        readiness=_Ready(),
        streaming_speech=_StreamingPort(),
    )
    headers = {"Authorization": f"Bearer {_KEY}", "Origin": "https://test.example"}

    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe", headers=headers) as first:
            assert first.receive_json() == {"type": "timeout"}
        with client.websocket_connect("/ws/transcribe", headers=headers) as second:
            assert second.receive_json() == {"type": "timeout"}


def test_deadline_releases_slot_before_blocking_analyze_and_close() -> None:
    app = create_app(
        settings=_timeout_settings(),
        analyzer=cast(AnalyzeCallPort, _BlockingAnalyzer()),
        readiness=_Ready(),
        streaming_speech=_BlockingStreamingPort(),
    )
    headers = {"Authorization": f"Bearer {_KEY}", "Origin": "https://test.example"}

    with TestClient(app) as client:
        with client.websocket_connect("/ws/transcribe", headers=headers) as websocket:
            websocket.send_json(
                {"type": "start", "sequence": 0, "codec": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1}
            )
            assert websocket.receive_json()["type"] == "started"
            websocket.send_json({"type": "audio", "sequence": 1, "data": base64.b64encode(b"pcm0").decode("ascii")})
            assert websocket.receive_json()["type"] == "partial"
            websocket.send_json({"type": "end", "sequence": 2})
            assert websocket.receive_json()["type"] == "provisional_final"
            assert websocket.receive_json() == {"type": "timeout"}
        assert app.state.ws_sessions._active == 0
