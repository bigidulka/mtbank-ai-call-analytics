"""Single-pass canonical batch pipeline from bytes to immutable TranscriptSnapshot."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import ASRMetadata, ASRProviderMetadata, TranscriptSegment, TranscriptSnapshot
from mtbank_ai.observability import Telemetry
from mtbank_ai.speech.contracts import (
    DiarizedSegment,
    SpeakerAttributedSegment,
    SpeechFile,
    SpeechTranscriptionResponse,
)
from mtbank_ai.speech.roles import RoleResolverPort, resolve_roles
from services.speech.errors import NoSpeechError, SpeechProviderError
from services.speech.manifest import ModelRegistry
from services.speech.media import NormalizedAudio
from services.speech.ports import SpeechPorts
from services.speech.settings import FasterWhisperSettings, SpeechRuntimeSettings

_UNASSIGNED_SPEAKER_ID = "UNASSIGNED"


class MediaNormalizerPort(Protocol):
    def normalize(self, source: SpeechFile) -> AbstractContextManager[NormalizedAudio]: ...


class CanonicalBatchEngine:
    """A request has exactly one local faster-whisper transcription invocation."""

    def __init__(
        self,
        *,
        normalizer: MediaNormalizerPort,
        ports: SpeechPorts,
        registry: ModelRegistry,
        runtime: SpeechRuntimeSettings,
        faster_whisper: FasterWhisperSettings,
        role_resolver: RoleResolverPort | None = None,
        clock: Callable[[], datetime] | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._normalizer = normalizer
        self._ports = ports
        self._registry = registry
        self._runtime = runtime
        self._faster_whisper = faster_whisper
        self._role_resolver = role_resolver
        self._clock = clock or (lambda: datetime.now(UTC))
        self._telemetry = telemetry or Telemetry()

    def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        started = time.perf_counter()
        with self._normalizer.normalize(source) as audio:
            # This is deliberately the only ASR invocation in the canonical path.
            with self._telemetry.span("speech.asr"):
                transcription = self._ports.transcriber.transcribe(audio, language=self._runtime.language)
            if not transcription.segments:
                raise NoSpeechError("local faster-whisper returned no speech segments")

            # Native faster-whisper word timestamps are already aligned.
            with self._telemetry.span("speech.alignment"):
                aligned = self._ports.aligner.align(audio, transcription, language=self._runtime.language)
            if not aligned:
                raise NoSpeechError("local faster-whisper returned no aligned speech segments")

            with self._telemetry.span("speech.diarization"):
                diarization = self._ports.diarizer.diarize(audio)
            with self._telemetry.span("speech.speaker_assignment"):
                attributed = self._ports.speaker_assigner.assign(aligned, diarization)
            if not attributed:
                raise NoSpeechError("speaker assignment returned no speech segments")
            if tuple(segment.start for segment in attributed) != tuple(sorted(segment.start for segment in attributed)):
                raise SpeechProviderError("speaker assignment returned non-monotonic timestamps")

            snapshot = self._snapshot(
                source,
                audio.audio_sha256,
                audio.duration_seconds,
                attributed,
                transcription.provider_metadata,
                started,
            )
        return SpeechTranscriptionResponse(transcript=snapshot)

    def _snapshot(
        self,
        source: SpeechFile,
        audio_sha256: str,
        duration_seconds: float,
        attributed: tuple[SpeakerAttributedSegment, ...],
        provider_metadata: ASRProviderMetadata | None,
        started: float,
    ) -> TranscriptSnapshot:
        diarized_segments = tuple(
            DiarizedSegment(
                id=_segment_id(audio_sha256, index, segment),
                original_speaker_id=segment.original_speaker_id or _UNASSIGNED_SPEAKER_ID,
                speaker_confidence=segment.speaker_confidence,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                word_timestamps=segment.words,
            )
            for index, segment in enumerate(attributed)
        )
        if not diarized_segments:
            raise NoSpeechError("speaker assignment returned no speech segments")

        with self._telemetry.span("speech.role_resolution"):
            resolution = resolve_roles(
                diarized_segments,
                metadata_mappings=source.metadata.role_mappings,
                resolver=self._role_resolver,
                review_confidence_threshold=self._runtime.role_review_confidence_threshold,
            )
        assignments = {item.original_speaker_id: item for item in resolution.assignments}
        resolved_segments = tuple(
            TranscriptSegment(
                id=segment.id,
                original_speaker_id=segment.original_speaker_id,
                speaker=assignments[segment.original_speaker_id].role,
                role_confidence=assignments[segment.original_speaker_id].confidence,
                speaker_confidence=segment.speaker_confidence,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                redacted_text=segment.text,
                word_timestamps=segment.word_timestamps,
            )
            for segment in diarized_segments
        )
        transcript_name = f"mtbank-ai/transcript/{audio_sha256}/{self._runtime.pipeline_revision}"
        return TranscriptSnapshot(
            transcript_id=uuid5(NAMESPACE_URL, transcript_name),
            audio_sha256=audio_sha256,
            revision=self._runtime.pipeline_revision,
            language=self._runtime.language,
            duration_seconds=duration_seconds,
            segments=resolved_segments,
            role_resolution=resolution,
            asr_metadata=ASRMetadata(
                asr=self._registry.asr_revision(),
                alignment=ComponentRevision(
                    package="faster-whisper",
                    package_version="1.2.1",
                    model_id=self._faster_whisper.model_id,
                    model_revision="native-word-timestamps-v1",
                    artifact_sha256=self._registry.manifest.asr.artifact_sha256,
                ),
                diarization=self._registry.diarization_revision(),
                language=self._runtime.language,
                processing_ms=int((time.perf_counter() - started) * 1000),
                provider=provider_metadata,
            ),
            created_at=self._clock(),
        )


def _segment_id(audio_sha256: str, index: int, segment: object) -> UUID:
    start = getattr(segment, "start")
    end = getattr(segment, "end")
    text = getattr(segment, "text")
    original_speaker_id = getattr(segment, "original_speaker_id") or _UNASSIGNED_SPEAKER_ID
    value = f"mtbank-ai/segment/{audio_sha256}/{index}/{original_speaker_id}/{start:.6f}/{end:.6f}/{text}"
    return uuid5(NAMESPACE_URL, value)
