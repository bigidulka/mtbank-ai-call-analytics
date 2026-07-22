"""Ports around heavy ASR/alignment/diarization libraries."""

from __future__ import annotations

from dataclasses import dataclass

from mtbank_ai.speech.contracts import (
    AlignedSegment,
    DiarizationTurn,
    SpeakerAttributedSegment,
    TranscriptionResult,
)
from services.speech.media import NormalizedAudio


class TranscriberPort:
    def transcribe(self, audio: NormalizedAudio, *, language: str) -> TranscriptionResult:
        raise NotImplementedError

    def warm(self) -> None:
        """Load local model state before an externally visible GPU readiness signal."""

        return None


class AlignerPort:
    def align(
        self,
        audio: NormalizedAudio,
        transcription: TranscriptionResult,
        *,
        language: str,
    ) -> tuple[AlignedSegment, ...]:
        raise NotImplementedError


class DiarizerPort:
    def diarize(self, audio: NormalizedAudio) -> tuple[DiarizationTurn, ...]:
        raise NotImplementedError

    def warm(self) -> None:
        """Load local model state before an externally visible GPU readiness signal."""

        return None


class SpeakerAssignerPort:
    def assign(
        self,
        aligned_segments: tuple[AlignedSegment, ...],
        diarization: tuple[DiarizationTurn, ...],
    ) -> tuple[SpeakerAttributedSegment, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SpeechPorts:
    transcriber: TranscriberPort
    aligner: AlignerPort
    diarizer: DiarizerPort
    speaker_assigner: SpeakerAssignerPort

    def warm(self) -> None:
        self.transcriber.warm()
        self.diarizer.warm()
