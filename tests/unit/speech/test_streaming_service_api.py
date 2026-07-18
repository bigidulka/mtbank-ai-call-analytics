from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import SecretStr

from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse
from mtbank_ai.speech.streaming import StreamingStart, StreamingUpdate
from services.speech.app import create_app
from services.speech.settings import (
    GroqTranscriptionSettings,
    SpeechRuntimeSettings,
    SpeechSettings,
    SpeechStreamingSettings,
)


class _Session:
    def __init__(self) -> None:
        self.closed = False

    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]:
        assert frame == b"\x01\x00"
        assert sequence == 1
        return (StreamingUpdate(sequence=sequence, text="частичный текст", stable_prefix=False),)

    async def finish(self) -> tuple[StreamingUpdate, ...]:
        return (StreamingUpdate(sequence=1, text="итоговый текст", stable_prefix=True, final=True),)

    async def close(self) -> None:
        self.closed = True


class _Runtime:
    def __init__(self) -> None:
        self.start: StreamingStart | None = None
        self.session = _Session()

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        del source
        raise AssertionError("batch transcription is not expected")

    async def open_stream(self, start: StreamingStart) -> _Session:
        self.start = start
        return self.session

    async def ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def test_internal_streaming_api_emits_unstable_first_partial_and_final(tmp_path) -> None:
    runtime = _Runtime()
    settings = SpeechSettings(
        runtime=SpeechRuntimeSettings(temp_root=str(tmp_path / "work")),
        groq=GroqTranscriptionSettings(api_key=SecretStr("test-groq-key")),
        streaming=SpeechStreamingSettings(enabled=True),
    )
    app = create_app(settings=settings, runtime=runtime)

    with TestClient(app).websocket_connect("/v1/stream") as websocket:
        websocket.send_json(
            {"type": "start", "sequence": 0, "codec": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1}
        )
        assert websocket.receive_json() == {"type": "started", "sequence": 0}
        websocket.send_bytes(b"\x00\x00\x00\x01\x01\x00")
        assert websocket.receive_json() == {
            "type": "update",
            "sequence": 1,
            "text": "частичный текст",
            "stable_prefix": False,
            "final": False,
        }
        assert websocket.receive_json() == {"type": "ack", "sequence": 1}
        websocket.send_json({"type": "end", "sequence": 2})
        assert websocket.receive_json() == {
            "type": "update",
            "sequence": 1,
            "text": "итоговый текст",
            "stable_prefix": True,
            "final": True,
        }
        assert websocket.receive_json() == {"type": "finished", "sequence": 2}

    assert runtime.start == StreamingStart("pcm_s16le", 16_000, 1)
    assert runtime.session.closed
