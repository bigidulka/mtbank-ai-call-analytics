from __future__ import annotations

import asyncio
from threading import Event
from typing import cast

import pytest

from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse
from mtbank_ai.speech.streaming import StreamingStart, StreamingUpdate
from services.speech import runtime as speech_runtime
from services.speech.errors import SpeechConfigurationError, SpeechOverloadedError
from services.speech.runtime import LazySpeechRuntime
from services.speech.settings import SpeechRuntimeSettings, SpeechStreamingSettings
from tests.unit.speech._helpers import make_registry


class BlockingEngine:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        del source
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=3.0)
        return cast(SpeechTranscriptionResponse, object())


class SuccessfulEngine:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        del source
        self.calls += 1
        return cast(SpeechTranscriptionResponse, object())


def test_runtime_is_lazy_and_rejects_requests_outside_bounded_queue(tmp_path) -> None:
    runtime_settings = SpeechRuntimeSettings(
        temp_root=str(tmp_path / "work"),
        request_timeout_seconds=5.0,
        queue_capacity=0,
    )
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    engine = BlockingEngine()
    factory_calls = 0

    def factory(registry, runtime, groq, resolver):
        nonlocal factory_calls
        del registry, runtime, groq, resolver
        factory_calls += 1
        return cast(object, engine)

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        assert await runtime.ready()
        assert factory_calls == 0
        first = asyncio.create_task(runtime.transcribe(SpeechFile("call.wav", "audio/wav", b"RIFF")))
        assert await asyncio.to_thread(engine.started.wait, 1.0)
        with pytest.raises(SpeechOverloadedError):
            await runtime.transcribe(SpeechFile("second.wav", "audio/wav", b"RIFF"))
        engine.release.set()
        await first
        assert engine.calls == 1
        assert factory_calls == 1

    asyncio.run(scenario())


def test_runtime_releases_slot_after_each_successful_request(tmp_path) -> None:
    runtime_settings = SpeechRuntimeSettings(
        temp_root=str(tmp_path / "work"),
        request_timeout_seconds=5.0,
        queue_capacity=0,
    )
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    engine = SuccessfulEngine()

    def factory(registry, runtime, groq, resolver):
        del registry, runtime, groq, resolver
        return cast(object, engine)

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        for index in range(3):
            await runtime.transcribe(SpeechFile(f"call-{index}.wav", "audio/wav", b"RIFF"))
        assert engine.calls == 3

    asyncio.run(scenario())


def test_cpu_readiness_cancellation_does_not_poison_future_probes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, settings = make_registry(tmp_path)
    started = Event()
    release = Event()

    def verify_ready() -> bool:
        started.set()
        assert release.wait(timeout=3.0)
        return True

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings)
        monkeypatch.setattr(runtime._registry, "verify_ready", verify_ready)
        task = asyncio.create_task(runtime.ready())
        assert await asyncio.to_thread(started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await runtime.ready()

    asyncio.run(scenario())


def test_cuda_readiness_is_status_only_after_successful_warmup(tmp_path) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    warm_calls = 0

    class WarmEngine:
        def warm(self) -> None:
            nonlocal warm_calls
            warm_calls += 1

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, WarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        assert not await runtime.ready()
        assert warm_calls == 0
        await runtime.warmup()
        assert await runtime.ready()
        assert warm_calls == 1

    asyncio.run(scenario())


def test_cuda_readiness_propagates_cancellation_and_fails_closed(tmp_path) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    started = Event()
    release = Event()

    class WarmEngine:
        def warm(self) -> None:
            started.set()
            assert release.wait(timeout=3.0)

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, WarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        task = asyncio.create_task(runtime.warmup())
        assert await asyncio.to_thread(started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert not await runtime.ready()

    asyncio.run(scenario())


def test_cuda_readiness_fails_closed_when_warmup_times_out(tmp_path) -> None:
    runtime_settings = SpeechRuntimeSettings(
        device="cuda",
        temp_root=str(tmp_path / "work"),
        request_timeout_seconds=0.01,
    )
    _, settings = make_registry(tmp_path, runtime=runtime_settings)
    started = Event()
    release = Event()

    class WarmEngine:
        def warm(self) -> None:
            started.set()
            assert release.wait(timeout=3.0)

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, WarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        with pytest.raises(SpeechConfigurationError):
            await runtime.warmup()
        assert await asyncio.to_thread(started.wait, 1.0)
        assert not await runtime.ready()
        release.set()

    asyncio.run(scenario())


def test_cuda_readiness_fails_closed_when_model_warmup_fails(tmp_path) -> None:
    runtime_settings = SpeechRuntimeSettings(device="cuda", temp_root=str(tmp_path / "work"))
    _, settings = make_registry(tmp_path, runtime=runtime_settings)

    class FailingWarmEngine:
        def warm(self) -> None:
            raise RuntimeError("model load failed")

    def factory(registry, runtime, faster_whisper, resolver):
        del registry, runtime, faster_whisper, resolver
        return cast(object, FailingWarmEngine())

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        with pytest.raises(SpeechConfigurationError):
            await runtime.warmup()
        assert not await runtime.ready()

    asyncio.run(scenario())


def test_runtime_rechecks_artifacts_before_first_adapter_graph_load(tmp_path) -> None:
    _, settings = make_registry(tmp_path)
    factory_calls = 0

    def factory(registry, runtime, groq, resolver):
        nonlocal factory_calls
        del registry, runtime, groq, resolver
        factory_calls += 1
        raise AssertionError("adapter graph must not load after readiness becomes false")

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings, engine_factory=factory)  # type: ignore[arg-type]
        assert await runtime.ready()
        (tmp_path / "artifacts" / "diarization" / "artifact.bin").write_bytes(b"tampered")

        with pytest.raises(SpeechConfigurationError):
            await runtime.transcribe(SpeechFile("call.wav", "audio/wav", b"RIFF"))
        assert factory_calls == 0

    asyncio.run(scenario())


class StreamingSession:
    def __init__(self) -> None:
        self.closed = False

    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]:
        del frame, sequence
        return ()

    async def finish(self) -> tuple[StreamingUpdate, ...]:
        return ()

    async def close(self) -> None:
        self.closed = True


class StreamingAdapter:
    def __init__(self) -> None:
        self.starts: list[StreamingStart] = []
        self.sessions: list[StreamingSession] = []

    async def open(self, start: StreamingStart) -> StreamingSession:
        self.starts.append(start)
        session = StreamingSession()
        self.sessions.append(session)
        return session


def test_streaming_runtime_wires_groq_settings_into_production_adapter(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, settings = make_registry(tmp_path)
    settings = settings.model_copy(update={"streaming": SpeechStreamingSettings(enabled=True)})
    captured: dict[str, object] = {}

    class CapturingAdapter:
        def __init__(self, registry, runtime, groq, streaming):
            captured.update(registry=registry, runtime=runtime, groq=groq, streaming=streaming)

        async def open(self, start: StreamingStart) -> StreamingSession:
            del start
            return StreamingSession()

    monkeypatch.setattr(speech_runtime, "ProductionStreamingSpeechAdapter", CapturingAdapter)

    async def scenario() -> None:
        runtime = LazySpeechRuntime(settings)
        session = await runtime.open_stream(StreamingStart("pcm_s16le", 16_000, 1))
        await session.close()

    asyncio.run(scenario())

    assert captured["groq"] is settings.groq
    assert captured["streaming"] is settings.streaming


def test_streaming_settings_validate_groq_rolling_budgets() -> None:
    assert SpeechStreamingSettings(enabled=True).enabled
    with pytest.raises(ValueError, match="rolling call timeout"):
        SpeechStreamingSettings(rolling_call_timeout_seconds=3.0, processing_timeout_seconds=2.0)
    with pytest.raises(ValueError, match="rolling audio budget"):
        SpeechStreamingSettings(max_rolling_audio_seconds_per_session=301.0)
