from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import cast

from pydantic import SecretStr

from mtbank_ai.speech.contracts import RecognizedSegment, TranscriptionResult
from mtbank_ai.speech.streaming import StreamingStart
from services.speech.adapters import GroqWhisperTranscriber
from services.speech.errors import SpeechProviderError
from services.speech.settings import GroqTranscriptionSettings, SpeechRuntimeSettings
from services.speech.streaming import GroqRollingTranscriber, RollingStreamingSession, StreamingRuntimeLimits


class _SyncGroq:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe(self, audio, *, language: str) -> TranscriptionResult:  # type: ignore[no-untyped-def]
        assert language == "ru"
        self.paths.append(audio.path)
        with wave.open(str(audio.path), "rb") as source:
            assert (source.getnchannels(), source.getframerate(), source.getsampwidth()) == (1, 16_000, 2)
            assert source.readframes(source.getnframes()) == b"\x01\x00" * 16
        return TranscriptionResult(
            language="ru",
            segments=(RecognizedSegment(start=0.0, end=0.001, text="временный wav"),),
        )


class _AsyncTranscriber:
    def __init__(self, results: tuple[str | Exception, ...]) -> None:
        self._results = iter(results)
        self.calls: list[bytes] = []

    async def transcribe(self, pcm_s16le: bytes) -> str:
        self.calls.append(pcm_s16le)
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


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


def test_streaming_runtime_has_no_local_asr_or_silero_dependency() -> None:
    root = Path(__file__).parents[3]
    source = (root / "services" / "speech" / "streaming.py").read_text(encoding="utf-8").casefold()
    project = (root / "services" / "speech" / "pyproject.toml").read_text(encoding="utf-8").casefold()

    for forbidden in ("silero", "onnxruntime", "faster_whisper", "whisperx"):
        assert forbidden not in source
        assert forbidden not in project


def test_groq_rolling_transcriber_writes_wav_calls_once_and_removes_temp(tmp_path: Path) -> None:
    fake = _SyncGroq()
    rolling = GroqRollingTranscriber(
        groq=GroqTranscriptionSettings(api_key=SecretStr("test-groq-key")),
        runtime=SpeechRuntimeSettings(temp_root=str(tmp_path)),
        rolling_timeout_seconds=1.0,
        semaphore=asyncio.Semaphore(1),
        transcriber=cast(GroqWhisperTranscriber, fake),
    )

    text = asyncio.run(rolling.transcribe(b"\x01\x00" * 16))

    assert text == "временный wav"
    assert len(fake.paths) == 1
    assert not fake.paths[0].exists()
    assert not tuple(tmp_path.glob("rolling-*.wav"))


def test_rolling_budget_and_provider_failure_skip_provisional_without_fallback() -> None:
    async def scenario() -> None:
        transcriber = _AsyncTranscriber((SpeechProviderError("provider failed"), "вторая попытка"))
        session = RollingStreamingSession(
            start=StreamingStart("pcm_s16le", 16_000, 1),
            limits=_limits(),
            transcriber=transcriber,
            rolling_window_seconds=1.0,
            rolling_step_seconds=1 / 16_000,
            rolling_call_timeout_seconds=1.0,
            max_rolling_calls_per_session=3,
            max_rolling_audio_seconds_per_session=1 / 16_000,
            pcm_energy_threshold=0,
            decoder=None,
        )

        assert await session.push(b"\x01\x00", sequence=1) == ()
        assert await session.push(b"\x01\x00", sequence=2) == ()
        assert await session.finish() == ()
        assert transcriber.calls == [b"\x01\x00"]

    asyncio.run(scenario())


def test_rolling_timeout_does_not_close_session() -> None:
    class _SlowTranscriber:
        def __init__(self) -> None:
            self.calls = 0

        async def transcribe(self, pcm_s16le: bytes) -> str:
            del pcm_s16le
            self.calls += 1
            await asyncio.sleep(0.01)
            return "слишком поздно"

    async def scenario() -> None:
        transcriber = _SlowTranscriber()
        session = RollingStreamingSession(
            start=StreamingStart("pcm_s16le", 16_000, 1),
            limits=_limits(),
            transcriber=transcriber,
            rolling_window_seconds=1.0,
            rolling_step_seconds=1 / 16_000,
            rolling_call_timeout_seconds=0.001,
            max_rolling_calls_per_session=2,
            max_rolling_audio_seconds_per_session=1.0,
            pcm_energy_threshold=0,
            decoder=None,
        )

        assert await session.push(b"\x01\x00", sequence=1) == ()
        assert await session.push(b"\x01\x00", sequence=2) == ()
        assert transcriber.calls == 2
        await session.finish()

    asyncio.run(scenario())
