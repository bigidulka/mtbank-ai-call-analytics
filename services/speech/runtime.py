"""Lazy one-device runtime with bounded admission and fail-closed readiness."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, cast

from mtbank_ai.policies import PolicyLoadError, PolicyRegistry
from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse
from mtbank_ai.speech.roles import PolicyRoleResolver, RoleResolverPort
from mtbank_ai.speech.streaming import StreamingSpeechSession, StreamingStart, StreamingUpdate
from services.speech.adapters import build_production_ports
from services.speech.engine import CanonicalBatchEngine
from services.speech.errors import (
    SpeechConfigurationError,
    SpeechDeadlineExceededError,
    SpeechOverloadedError,
    SpeechProviderError,
)
from services.speech.manifest import ModelRegistry
from services.speech.media import MediaLimits, MediaNormalizer
from services.speech.settings import GroqTranscriptionSettings, SpeechRuntimeSettings, SpeechSettings
from services.speech.streaming import ProductionStreamingSpeechAdapter


class SpeechRuntimePort(Protocol):
    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse: ...

    async def ready(self) -> bool: ...

    async def close(self) -> None: ...


class StreamingRuntimePort(Protocol):
    async def open_stream(self, start: StreamingStart) -> StreamingSpeechSession: ...


EngineFactory = Callable[
    [ModelRegistry, SpeechRuntimeSettings, GroqTranscriptionSettings, RoleResolverPort | None],
    CanonicalBatchEngine,
]


class _UseDefaultRoleResolver:
    pass


_DEFAULT_ROLE_RESOLVER = _UseDefaultRoleResolver()


class LazySpeechRuntime:
    """One app process owns one lazily-created adapter graph for its configured device."""

    def __init__(
        self,
        settings: SpeechSettings,
        *,
        engine_factory: EngineFactory | None = None,
        role_resolver: RoleResolverPort | None | _UseDefaultRoleResolver = _DEFAULT_ROLE_RESOLVER,
    ) -> None:
        self._settings = settings
        self._registry = ModelRegistry.load(settings)
        self._engine_factory = engine_factory or _production_engine
        if role_resolver is _DEFAULT_ROLE_RESOLVER:
            self._role_resolver = _load_default_role_resolver()
        else:
            self._role_resolver = cast(RoleResolverPort | None, role_resolver)
        self._engine: CanonicalBatchEngine | None = None
        self._engine_lock = asyncio.Lock()
        self._streaming: ProductionStreamingSpeechAdapter | None = None
        self._streaming_lock = asyncio.Lock()
        self._model_slot = asyncio.Semaphore(settings.runtime.max_concurrency)
        self._admission_lock = asyncio.Lock()
        self._outstanding = 0
        self._fatal_provider_failure = False

    async def ready(self) -> bool:
        if self._fatal_provider_failure:
            return False
        return await asyncio.to_thread(self._registry.verify_ready)

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        await self._reserve_slot()
        worker = asyncio.create_task(self._execute(source))
        try:
            result = await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=self._settings.runtime.request_timeout_seconds,
            )
        except TimeoutError:
            self._release_when_finished(worker)
            raise SpeechDeadlineExceededError("canonical speech request exceeded deadline") from None
        except BaseException:
            self._release_when_finished(worker)
            raise
        await self._release_slot()
        return result

    async def open_stream(self, start: StreamingStart) -> StreamingSpeechSession:
        if not self._settings.streaming.enabled:
            raise SpeechConfigurationError("streaming speech is disabled")
        await self._reserve_slot()
        acquired_model_slot = False
        try:
            await self._model_slot.acquire()
            acquired_model_slot = True
            adapter = await self._get_streaming_adapter()
            session = await adapter.open(start)
            return _ReservedStreamingSession(session, self._model_slot, self._release_slot)
        except BaseException:
            if acquired_model_slot:
                self._model_slot.release()
            await self._release_slot()
            raise

    async def close(self) -> None:
        return None

    async def _execute(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        async with self._model_slot:
            engine = await self._get_engine()
            try:
                return await asyncio.to_thread(engine.transcribe, source)
            except SpeechProviderError:
                self._fatal_provider_failure = True
                raise

    async def _get_engine(self) -> CanonicalBatchEngine:
        async with self._engine_lock:
            if self._engine is None:
                if not await asyncio.to_thread(self._registry.verify_ready):
                    raise SpeechConfigurationError("local speech model artifacts are not ready")
                self._engine = self._engine_factory(
                    self._registry,
                    self._settings.runtime,
                    self._settings.groq,
                    self._role_resolver,
                )
            return self._engine

    async def _get_streaming_adapter(self) -> ProductionStreamingSpeechAdapter:
        async with self._streaming_lock:
            if self._streaming is None:
                if not await asyncio.to_thread(self._registry.verify_ready):
                    raise SpeechConfigurationError("local speech model artifacts are not ready")
                self._streaming = ProductionStreamingSpeechAdapter(
                    self._registry,
                    self._settings.runtime,
                    self._settings.groq,
                    self._settings.streaming,
                )
            return self._streaming

    async def _reserve_slot(self) -> None:
        async with self._admission_lock:
            capacity = self._settings.runtime.max_concurrency + self._settings.runtime.queue_capacity
            if self._outstanding >= capacity:
                raise SpeechOverloadedError("speech queue is full")
            self._outstanding += 1

    async def _release_slot(self) -> None:
        async with self._admission_lock:
            self._outstanding -= 1

    def _release_when_finished(self, worker: asyncio.Task[SpeechTranscriptionResponse]) -> None:
        if worker.done():
            asyncio.create_task(self._release_slot())
            return
        worker.add_done_callback(lambda _: asyncio.create_task(self._release_slot()))


class _ReservedStreamingSession:
    def __init__(
        self,
        session: StreamingSpeechSession,
        model_slot: asyncio.Semaphore,
        release_admission: Callable[[], Awaitable[None]],
    ) -> None:
        self._session = session
        self._model_slot = model_slot
        self._release_admission = release_admission
        self._closed = False

    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]:
        return await self._session.push(frame, sequence=sequence)

    async def finish(self) -> tuple[StreamingUpdate, ...]:
        try:
            return await self._session.finish()
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._session.close()
        finally:
            self._model_slot.release()
            await self._release_admission()


class UnavailableSpeechRuntime:
    """Keeps liveness available while readiness and requests fail closed."""

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        del source
        raise SpeechConfigurationError("speech runtime configuration is unavailable")

    async def open_stream(self, start: StreamingStart) -> StreamingSpeechSession:
        del start
        raise SpeechConfigurationError("streaming speech runtime configuration is unavailable")

    async def ready(self) -> bool:
        return False

    async def close(self) -> None:
        return None


def _load_default_role_resolver() -> PolicyRoleResolver:
    try:
        return PolicyRoleResolver(PolicyRegistry().roles)
    except (PolicyLoadError, ValueError) as error:
        raise SpeechConfigurationError("verified roles policy is unavailable") from error


def _production_engine(
    registry: ModelRegistry,
    runtime: SpeechRuntimeSettings,
    groq: GroqTranscriptionSettings,
    role_resolver: RoleResolverPort | None,
) -> CanonicalBatchEngine:
    return CanonicalBatchEngine(
        normalizer=MediaNormalizer(
            MediaLimits(
                max_upload_bytes=runtime.max_upload_bytes,
                max_duration_seconds=runtime.max_duration_seconds,
                process_timeout_seconds=runtime.ffmpeg_timeout_seconds,
                temp_root=Path(runtime.temp_root),
                sample_rate_hz=runtime.normalization_sample_rate_hz,
                channels=runtime.normalization_channels,
                codec=runtime.normalization_codec,
            )
        ),
        ports=build_production_ports(registry, runtime, groq),
        registry=registry,
        runtime=runtime,
        groq=groq,
        role_resolver=role_resolver,
    )
