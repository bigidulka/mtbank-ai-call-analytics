"""Production-only bounded VAD, rolling ASR, and Ogg/Opus decode runtime."""

from __future__ import annotations

import asyncio
import hashlib
import os
import signal
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mtbank_ai.speech.streaming import (
    StreamingAdapterUnavailable,
    StreamingSpeechSession,
    StreamingStart,
    StreamingUpdate,
    validate_stream_frame,
)
from services.speech.adapters import GroqWhisperTranscriber
from services.speech.errors import SpeechProviderError
from services.speech.manifest import ModelRegistry
from services.speech.media import NormalizedAudio
from services.speech.settings import GroqTranscriptionSettings, SpeechRuntimeSettings, SpeechStreamingSettings

_PCM_SAMPLE_BYTES = 2
_PCM_SAMPLE_RATE_HZ = 16_000
_PCM_BYTES_PER_SECOND = _PCM_SAMPLE_RATE_HZ * _PCM_SAMPLE_BYTES
_OGG_HEADER_MAX_BYTES = 8 * 1024
_OGG_CAPTURE_PATTERN = b"OggS"
_OGG_FIXED_HEADER_BYTES = 27
_OGG_MAX_SEGMENTS = 255
_OGG_MAX_PAGE_BYTES = _OGG_FIXED_HEADER_BYTES + _OGG_MAX_SEGMENTS + _OGG_MAX_SEGMENTS * _OGG_MAX_SEGMENTS
_OGG_VALID_HEADER_TYPE_MASK = 0x07
_OGG_BOS_FLAG = 0x02
_OGG_EOS_FLAG = 0x04
_FFMPEG_READ_BYTES = 16 * 1024


class RollingTranscriberPort(Protocol):
    async def transcribe(self, pcm_s16le: bytes) -> str: ...


@dataclass(frozen=True, slots=True)
class StreamingRuntimeLimits:
    max_frame_bytes: int
    max_session_bytes: int
    max_duration_seconds: float
    processing_timeout_seconds: float
    ffmpeg_timeout_seconds: float
    max_decoder_output_bytes: int
    max_decoder_stderr_bytes: int


class _OggLogicalStreamValidator:
    """Accepts one contiguous Ogg logical stream before it reaches FFmpeg."""

    def __init__(self, max_frame_bytes: int) -> None:
        self._pending = bytearray()
        self._max_pending_bytes = max_frame_bytes + max(max_frame_bytes, _OGG_MAX_PAGE_BYTES)
        self._serial_number: int | None = None
        self._expected_page_sequence: int | None = None
        self._page_count = 0
        self._saw_eos = False

    def feed(self, payload: bytes) -> bytes:
        if self._saw_eos or len(self._pending) + len(payload) > self._max_pending_bytes:
            raise StreamingAdapterUnavailable("Ogg/Opus logical stream exceeded protocol bound")
        self._pending.extend(payload)
        validated = bytearray()
        while True:
            if len(self._pending) < _OGG_FIXED_HEADER_BYTES:
                return bytes(validated)
            if not self._pending.startswith(_OGG_CAPTURE_PATTERN):
                raise StreamingAdapterUnavailable("Ogg/Opus logical stream is malformed")
            version = self._pending[4]
            header_type = self._pending[5]
            segment_count = self._pending[26]
            header_bytes = _OGG_FIXED_HEADER_BYTES + segment_count
            if version != 0 or header_type & ~_OGG_VALID_HEADER_TYPE_MASK:
                raise StreamingAdapterUnavailable("Ogg/Opus logical stream is malformed")
            if len(self._pending) < header_bytes:
                return bytes(validated)
            page_bytes = header_bytes + sum(self._pending[_OGG_FIXED_HEADER_BYTES:header_bytes])
            if page_bytes > _OGG_MAX_PAGE_BYTES:
                raise StreamingAdapterUnavailable("Ogg/Opus page exceeded protocol bound")
            if len(self._pending) < page_bytes:
                return bytes(validated)
            page = bytes(self._pending[:page_bytes])
            serial_number = int.from_bytes(page[14:18], "little")
            page_sequence = int.from_bytes(page[18:22], "little")
            self._validate_page(serial_number, page_sequence, header_type, page[header_bytes:])
            del self._pending[:page_bytes]
            validated.extend(page)
            if self._saw_eos:
                if self._pending:
                    raise StreamingAdapterUnavailable("Ogg/Opus logical stream is discontinuous")
                return bytes(validated)

    def finish(self) -> None:
        if self._pending or self._page_count == 0 or not self._saw_eos:
            raise StreamingAdapterUnavailable("Ogg/Opus logical stream is incomplete")

    def _validate_page(
        self,
        serial_number: int,
        page_sequence: int,
        header_type: int,
        page_body: bytes,
    ) -> None:
        if self._page_count == 0:
            if (
                (header_type & _OGG_BOS_FLAG) == 0
                or (header_type & 0x01)
                or page_sequence != 0
                or not page_body.startswith(b"OpusHead")
            ):
                raise StreamingAdapterUnavailable("Ogg/Opus logical stream has invalid first page")
            self._serial_number = serial_number
        elif (
            header_type & _OGG_BOS_FLAG
            or serial_number != self._serial_number
            or page_sequence != self._expected_page_sequence
        ):
            raise StreamingAdapterUnavailable("Ogg/Opus logical stream is discontinuous")
        self._page_count += 1
        self._expected_page_sequence = (page_sequence + 1) % (1 << 32)
        if header_type & _OGG_EOS_FLAG:
            self._saw_eos = True


class PersistentOggOpusDecoder:
    """Single bounded FFmpeg process per Ogg/Opus logical stream, never a shell pipeline."""

    def __init__(self, limits: StreamingRuntimeLimits) -> None:
        self._limits = limits
        self._process: asyncio.subprocess.Process | None = None
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._input_bytes = 0
        self._decoded_output_bytes = 0
        self._closed = False

    @property
    def argv(self) -> tuple[str, ...]:
        return (
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-xerror",
            "-fflags",
            "+nobuffer",
            "-f",
            "ogg",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        )

    async def feed(self, payload: bytes) -> bytes:
        if self._closed or not payload:
            raise StreamingAdapterUnavailable("Ogg/Opus decoder is closed or received empty input")
        if self._input_bytes + len(payload) > self._limits.max_session_bytes:
            raise StreamingAdapterUnavailable("Ogg/Opus decoder input exceeded bound")
        await self._start()
        self._input_bytes += len(payload)
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(payload)
        try:
            await asyncio.wait_for(self._process.stdin.drain(), timeout=self._limits.processing_timeout_seconds)
        except (BrokenPipeError, TimeoutError) as error:
            await self.close()
            raise StreamingAdapterUnavailable("Ogg/Opus decoder backpressure exceeded") from error
        await asyncio.sleep(0)
        self._raise_if_failed()
        return self._take_stdout()

    async def finish(self) -> bytes:
        if self._closed:
            return b""
        await self._start()
        assert self._process is not None
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=self._limits.ffmpeg_timeout_seconds)
            await self._await_pumps()
            self._raise_if_failed()
            return self._take_stdout()
        except TimeoutError as error:
            await self.close()
            raise StreamingAdapterUnavailable("Ogg/Opus decoder exceeded deadline") from error
        finally:
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.returncode is None:
            _kill_process_group(process)
            try:
                await asyncio.wait_for(process.wait(), timeout=self._limits.processing_timeout_seconds)
            except TimeoutError:
                pass
        for task in (self._stdout_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
        await self._await_pumps()

    async def _start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=self._limits.max_decoder_output_bytes,
                start_new_session=True,
            )
        except OSError as error:
            raise StreamingAdapterUnavailable("FFmpeg streaming decoder is unavailable") from error
        assert self._process.stdout is not None and self._process.stderr is not None
        self._stdout_task = asyncio.create_task(
            self._pump(
                self._process.stdout,
                self._stdout,
                max_bytes=self._limits.max_decoder_output_bytes,
                count_decoded_output=True,
            )
        )
        self._stderr_task = asyncio.create_task(
            self._pump(
                self._process.stderr,
                self._stderr,
                max_bytes=self._limits.max_decoder_stderr_bytes,
                count_decoded_output=False,
            )
        )

    async def _pump(
        self,
        reader: asyncio.StreamReader,
        target: bytearray,
        *,
        max_bytes: int,
        count_decoded_output: bool,
    ) -> None:
        while chunk := await reader.read(_FFMPEG_READ_BYTES):
            if len(target) + len(chunk) > max_bytes or (
                count_decoded_output and self._decoded_output_bytes + len(chunk) > max_bytes
            ):
                process = self._process
                if process is not None:
                    _kill_process_group(process)
                raise StreamingAdapterUnavailable("FFmpeg streaming output exceeded bound")
            if count_decoded_output:
                self._decoded_output_bytes += len(chunk)
            target.extend(chunk)

    async def _await_pumps(self) -> None:
        for task in (self._stdout_task, self._stderr_task):
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, StreamingAdapterUnavailable):
                continue

    def _take_stdout(self) -> bytes:
        result = bytes(self._stdout)
        self._stdout.clear()
        return result

    def _raise_if_failed(self) -> None:
        process = self._process
        if process is not None and process.returncode not in (None, 0):
            raise StreamingAdapterUnavailable("FFmpeg rejected Ogg/Opus logical stream")


class GroqRollingTranscriber(RollingTranscriberPort):
    """Writes bounded PCM to a temporary WAV and invokes Groq once for each rolling call."""

    def __init__(
        self,
        *,
        groq: GroqTranscriptionSettings,
        runtime: SpeechRuntimeSettings,
        rolling_timeout_seconds: float,
        semaphore: asyncio.Semaphore,
        transcriber: GroqWhisperTranscriber | None = None,
    ) -> None:
        self._runtime = runtime
        self._semaphore = semaphore
        rolling_settings = groq.model_copy(
            update={
                "request_timeout_seconds": rolling_timeout_seconds,
                "connect_timeout_seconds": min(groq.connect_timeout_seconds, rolling_timeout_seconds),
            }
        )
        self._transcriber = transcriber or GroqWhisperTranscriber(rolling_settings)

    async def transcribe(self, pcm_s16le: bytes) -> str:
        if not pcm_s16le or len(pcm_s16le) % _PCM_SAMPLE_BYTES:
            return ""
        async with self._semaphore:
            return await asyncio.to_thread(self._transcribe_sync, pcm_s16le)

    def _transcribe_sync(self, pcm_s16le: bytes) -> str:
        root = Path(self._runtime.temp_root)
        path: Path | None = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix="rolling-", suffix=".wav", dir=root)
            path = Path(name)
            with os.fdopen(descriptor, "wb") as destination:
                with wave.open(destination, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(_PCM_SAMPLE_BYTES)
                    wav.setframerate(_PCM_SAMPLE_RATE_HZ)
                    wav.writeframes(pcm_s16le)
            result = self._transcriber.transcribe(
                NormalizedAudio(
                    path=path,
                    duration_seconds=len(pcm_s16le) / _PCM_BYTES_PER_SECOND,
                    audio_sha256=hashlib.sha256(pcm_s16le).hexdigest(),
                    source_format="wav",
                ),
                language=self._runtime.language,
            )
            return " ".join(segment.text for segment in result.segments).strip()
        except SpeechProviderError:
            raise
        except OSError as error:
            raise SpeechProviderError("Groq rolling transcription failed") from error
        finally:
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


class ProductionStreamingSpeechAdapter:
    """Creates bounded Groq-only rolling sessions; canonical reconciliation remains batch."""

    def __init__(
        self,
        registry: ModelRegistry,
        runtime: SpeechRuntimeSettings,
        groq: GroqTranscriptionSettings,
        settings: SpeechStreamingSettings,
        *,
        transcriber: RollingTranscriberPort | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        semaphore = asyncio.Semaphore(settings.max_concurrent_rolling_calls)
        self._transcriber = transcriber or GroqRollingTranscriber(
            groq=groq,
            runtime=runtime,
            rolling_timeout_seconds=settings.rolling_call_timeout_seconds,
            semaphore=semaphore,
        )

    async def open(self, start: StreamingStart) -> StreamingSpeechSession:
        if not self._settings.enabled:
            raise StreamingAdapterUnavailable("streaming speech is disabled")
        if not await asyncio.to_thread(self._registry.verify_ready):
            raise StreamingAdapterUnavailable("verified speech artifacts are unavailable")
        decoder = PersistentOggOpusDecoder(_runtime_limits(self._settings)) if start.codec == "ogg_opus" else None
        return RollingStreamingSession(
            start=start,
            limits=_runtime_limits(self._settings),
            transcriber=self._transcriber,
            rolling_window_seconds=self._settings.rolling_window_seconds,
            rolling_step_seconds=self._settings.rolling_step_seconds,
            rolling_call_timeout_seconds=self._settings.rolling_call_timeout_seconds,
            max_rolling_calls_per_session=self._settings.max_rolling_calls_per_session,
            max_rolling_audio_seconds_per_session=self._settings.max_rolling_audio_seconds_per_session,
            pcm_energy_threshold=self._settings.pcm_energy_threshold,
            max_update_text_bytes=self._settings.max_update_text_bytes,
            decoder=decoder,
        )


class RollingStreamingSession:
    """Fixed-cadence bounded PCM ring with Groq-only provisional updates."""

    def __init__(
        self,
        *,
        start: StreamingStart,
        limits: StreamingRuntimeLimits,
        transcriber: RollingTranscriberPort,
        rolling_window_seconds: float,
        rolling_step_seconds: float,
        rolling_call_timeout_seconds: float,
        max_rolling_calls_per_session: int,
        max_rolling_audio_seconds_per_session: float,
        pcm_energy_threshold: int,
        decoder: PersistentOggOpusDecoder | None,
        max_update_text_bytes: int = 48 * 1024,
    ) -> None:
        if (start.codec == "ogg_opus") != (decoder is not None):
            raise ValueError("Ogg/Opus streaming session requires a persistent decoder")
        self._start = start
        self._limits = limits
        self._transcriber = transcriber
        self._rolling_window_bytes = int(rolling_window_seconds * _PCM_BYTES_PER_SECOND)
        self._rolling_step_bytes = int(rolling_step_seconds * _PCM_BYTES_PER_SECOND)
        self._rolling_call_timeout_seconds = rolling_call_timeout_seconds
        self._max_rolling_calls = max_rolling_calls_per_session
        self._max_rolling_audio_seconds = max_rolling_audio_seconds_per_session
        self._pcm_energy_threshold = pcm_energy_threshold
        self._max_update_text_bytes = max_update_text_bytes
        self._decoder = decoder
        self._ogg_validator = _OggLogicalStreamValidator(limits.max_frame_bytes) if start.codec == "ogg_opus" else None
        self._speech_ring = bytearray()
        self._received_pcm_bytes = 0
        self._last_rolling_bytes = 0
        self._rolling_calls = 0
        self._rolling_audio_seconds = 0.0
        self._rolling_exhausted = False
        self._last_sequence = 0
        self._bytes_received = 0
        self._first_frame = True
        self._previous_tokens: tuple[str, ...] = ()
        self._committed_tokens: tuple[str, ...] = ()
        self._latest_text = ""
        self._closed = False
        self._opened_at = time.monotonic()
        self._header = bytearray()
        self._ogg_header_verified = start.codec != "ogg_opus"

    async def push(self, frame: bytes, *, sequence: int) -> tuple[StreamingUpdate, ...]:
        self._ensure_within_duration()
        if self._closed or sequence != self._last_sequence + 1:
            raise StreamingAdapterUnavailable("streaming session sequence violation")
        validate_stream_frame(self._start, frame, first_frame=self._first_frame)
        self._first_frame = False
        self._last_sequence = sequence
        self._bytes_received += len(frame)
        if len(frame) > self._limits.max_frame_bytes or self._bytes_received > self._limits.max_session_bytes:
            raise StreamingAdapterUnavailable("streaming session byte bound exceeded")
        return await self._consume_pcm(await self._decode(frame), sequence, force=False)

    async def finish(self) -> tuple[StreamingUpdate, ...]:
        if self._closed:
            return ()
        try:
            self._ensure_within_duration()
            pcm = await self._finish_decoder()
            updates = list(await self._consume_pcm(pcm, self._last_sequence, force=True))
            if not pcm and self._received_pcm_bytes > self._last_rolling_bytes:
                self._last_rolling_bytes = self._received_pcm_bytes
                updates.extend(await self._rolling_update(self._last_sequence))
            if self._latest_text:
                if updates and updates[-1].text == self._latest_text:
                    latest = updates[-1]
                    updates[-1] = StreamingUpdate(
                        sequence=latest.sequence,
                        text=latest.text,
                        stable_prefix=latest.stable_prefix,
                        final=True,
                    )
                else:
                    updates.append(
                        StreamingUpdate(
                            sequence=max(1, self._last_sequence),
                            text=self._latest_text,
                            stable_prefix=bool(self._committed_tokens),
                            final=True,
                        )
                    )
            return tuple(updates)
        finally:
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._decoder is not None:
            await self._decoder.close()

    def _ensure_within_duration(self) -> None:
        if time.monotonic() - self._opened_at > self._limits.max_duration_seconds:
            raise StreamingAdapterUnavailable("streaming session exceeded duration bound")

    async def _decode(self, frame: bytes) -> bytes:
        if self._decoder is None:
            return frame
        assert self._ogg_validator is not None
        frame = self._ogg_validator.feed(frame)
        if not frame:
            return b""
        if not self._ogg_header_verified:
            self._header.extend(frame)
            if b"OpusHead" not in self._header:
                if len(self._header) > _OGG_HEADER_MAX_BYTES:
                    raise StreamingAdapterUnavailable("Ogg/Opus identification header exceeded bound")
                return b""
            self._ogg_header_verified = True
            frame = bytes(self._header)
            self._header.clear()
        return await self._decoder.feed(frame)

    async def _finish_decoder(self) -> bytes:
        if self._decoder is None:
            return b""
        assert self._ogg_validator is not None
        self._ogg_validator.finish()
        if not self._ogg_header_verified:
            raise StreamingAdapterUnavailable("Ogg/Opus logical stream lacks OpusHead")
        return await self._decoder.finish()

    async def _consume_pcm(self, pcm: bytes, sequence: int, *, force: bool) -> tuple[StreamingUpdate, ...]:
        if len(pcm) % _PCM_SAMPLE_BYTES:
            raise StreamingAdapterUnavailable("decoded PCM contains partial sample")
        if not pcm:
            return ()
        self._received_pcm_bytes += len(pcm)
        self._speech_ring.extend(pcm)
        if len(self._speech_ring) > self._rolling_window_bytes:
            del self._speech_ring[: len(self._speech_ring) - self._rolling_window_bytes]
        ready = self._received_pcm_bytes - self._last_rolling_bytes >= self._rolling_step_bytes
        if not force and not ready:
            return ()
        self._last_rolling_bytes = self._received_pcm_bytes
        return await self._rolling_update(sequence)

    async def _rolling_update(self, sequence: int) -> tuple[StreamingUpdate, ...]:
        pcm = bytes(self._speech_ring)
        if (
            self._rolling_exhausted
            or not pcm
            or not _has_pcm_energy(pcm, self._pcm_energy_threshold)
        ):
            return ()
        duration_seconds = len(pcm) / _PCM_BYTES_PER_SECOND
        if (
            self._rolling_calls >= self._max_rolling_calls
            or self._rolling_audio_seconds + duration_seconds > self._max_rolling_audio_seconds
        ):
            self._rolling_exhausted = True
            return ()
        self._rolling_calls += 1
        self._rolling_audio_seconds += duration_seconds
        try:
            text = await asyncio.wait_for(
                self._transcriber.transcribe(pcm),
                timeout=self._rolling_call_timeout_seconds,
            )
        except (SpeechProviderError, TimeoutError):
            return ()
        return self._stable_updates(text, sequence)

    def _stable_updates(self, text: str, sequence: int) -> tuple[StreamingUpdate, ...]:
        normalized = " ".join(text.split())
        if not normalized:
            return ()
        if len(normalized.encode("utf-8")) > self._max_update_text_bytes:
            raise StreamingAdapterUnavailable("rolling ASR update exceeded bound")
        tokens = tuple(normalized.split(" "))
        if not self._previous_tokens:
            self._previous_tokens = tokens
            self._latest_text = normalized
            return (StreamingUpdate(sequence=max(1, sequence), text=normalized, stable_prefix=False),)
        stable = _common_prefix(self._previous_tokens, tokens)
        self._previous_tokens = tokens
        self._latest_text = normalized
        if len(stable) <= len(self._committed_tokens):
            return ()
        self._committed_tokens = stable
        return (StreamingUpdate(sequence=max(1, sequence), text=" ".join(stable), stable_prefix=True),)


def _runtime_limits(settings: SpeechStreamingSettings) -> StreamingRuntimeLimits:
    return StreamingRuntimeLimits(
        max_frame_bytes=settings.max_frame_bytes,
        max_session_bytes=settings.max_session_bytes,
        max_duration_seconds=settings.max_duration_seconds,
        processing_timeout_seconds=settings.processing_timeout_seconds,
        ffmpeg_timeout_seconds=settings.ffmpeg_timeout_seconds,
        max_decoder_output_bytes=settings.max_decoder_output_bytes,
        max_decoder_stderr_bytes=settings.max_decoder_stderr_bytes,
    )


def _has_pcm_energy(pcm_s16le: bytes, threshold: int) -> bool:
    return any(
        abs(int.from_bytes(pcm_s16le[index : index + _PCM_SAMPLE_BYTES], "little", signed=True)) >= threshold
        for index in range(0, len(pcm_s16le), _PCM_SAMPLE_BYTES)
    )


def _common_prefix(previous: tuple[str, ...], current: tuple[str, ...]) -> tuple[str, ...]:
    prefix: list[str] = []
    for old, new in zip(previous, current, strict=False):
        if old != new:
            break
        prefix.append(old)
    return tuple(prefix)


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
