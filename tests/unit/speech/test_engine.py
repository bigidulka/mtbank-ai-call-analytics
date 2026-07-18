from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mtbank_ai.domain.transcript import SpeakerRole, WordTimestamp
from mtbank_ai.speech.contracts import (
    AlignedSegment,
    DiarizationTurn,
    RecognizedSegment,
    SpeakerAttributedSegment,
    SpeakerRoleMapping,
    SpeechFile,
    SpeechMetadata,
    TranscriptionResult,
)
from mtbank_ai.speech.roles import RoleResolutionRequiredError
from services.speech.engine import CanonicalBatchEngine
from services.speech.errors import NoSpeechError
from services.speech.media import NormalizedAudio
from services.speech.ports import AlignerPort, DiarizerPort, SpeakerAssignerPort, SpeechPorts, TranscriberPort
from tests.unit.speech._helpers import make_registry


class FakeNormalizer:
    @contextmanager
    def normalize(self, source: SpeechFile) -> Generator[NormalizedAudio, None, None]:
        del source
        yield NormalizedAudio(
            path=Path("/tmp/fake-normalized.wav"),
            duration_seconds=2.0,
            audio_sha256="a" * 64,
            source_format="wav",
        )


class FakeTranscriber(TranscriberPort):
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls = 0

    def transcribe(self, audio: NormalizedAudio, *, language: str) -> TranscriptionResult:
        del audio
        assert language == "ru"
        self.calls += 1
        return self.result


class FakeAligner(AlignerPort):
    def __init__(self) -> None:
        self.calls = 0
        self.seen_transcription: TranscriptionResult | None = None

    def align(
        self,
        audio: NormalizedAudio,
        transcription: TranscriptionResult,
        *,
        language: str,
    ) -> tuple[AlignedSegment, ...]:
        del audio
        assert language == "ru"
        self.calls += 1
        self.seen_transcription = transcription
        return (
            AlignedSegment(
                start=0.0,
                end=1.0,
                text="Добрый день.",
                words=(WordTimestamp(word="Добрый", start=0.0, end=0.4, confidence=0.9),),
            ),
        )


class FakeDiarizer(DiarizerPort):
    def __init__(self) -> None:
        self.calls = 0

    def diarize(self, audio: NormalizedAudio) -> tuple[DiarizationTurn, ...]:
        del audio
        self.calls += 1
        return (DiarizationTurn(original_speaker_id="SPEAKER_00", start=0.0, end=1.0, confidence=0.8),)


class FakeSpeakerAssigner(SpeakerAssignerPort):
    def __init__(self) -> None:
        self.calls = 0

    def assign(
        self,
        aligned_segments: tuple[AlignedSegment, ...],
        diarization: tuple[DiarizationTurn, ...],
    ) -> tuple[SpeakerAttributedSegment, ...]:
        self.calls += 1
        assert len(aligned_segments) == len(diarization) == 1
        segment = aligned_segments[0]
        return (
            SpeakerAttributedSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=segment.words,
                original_speaker_id="SPEAKER_00",
                speaker_confidence=0.8,
            ),
        )


def test_canonical_engine_calls_full_transcription_once_and_reuses_it_for_alignment(tmp_path: Path) -> None:
    registry, settings = make_registry(tmp_path)
    transcription = FakeTranscriber(
        TranscriptionResult(
            language="ru",
            segments=(RecognizedSegment(start=0.0, end=1.0, text="Добрый день."),),
        )
    )
    aligner = FakeAligner()
    diarizer = FakeDiarizer()
    assigner = FakeSpeakerAssigner()
    engine = CanonicalBatchEngine(
        normalizer=FakeNormalizer(),
        ports=SpeechPorts(transcription, aligner, diarizer, assigner),
        registry=registry,
        runtime=settings.runtime,
        groq=settings.groq,
        clock=lambda: datetime(2026, 7, 15, tzinfo=UTC),
    )

    response = engine.transcribe(
        SpeechFile(
            filename="call.wav",
            content_type="audio/wav",
            content=b"RIFF",
            metadata=SpeechMetadata(
                role_mappings=(
                    SpeakerRoleMapping(
                        original_speaker_id="SPEAKER_00",
                        role=SpeakerRole.OPERATOR,
                        evidence="trusted-metadata",
                    ),
                )
            ),
        )
    )

    assert transcription.calls == aligner.calls == diarizer.calls == assigner.calls == 1
    assert aligner.seen_transcription is transcription.result
    assert response.transcript.segments[0].speaker is SpeakerRole.OPERATOR
    assert response.transcript.segments[0].speaker_confidence == 0.8
    assert not response.transcript.role_resolution.needs_review
    assert response.transcript.audio_sha256 == "a" * 64


def test_engine_returns_typed_pre_resolution_outcome_without_snapshot_for_unmapped_speaker(tmp_path: Path) -> None:
    registry, settings = make_registry(tmp_path)
    engine = CanonicalBatchEngine(
        normalizer=FakeNormalizer(),
        ports=SpeechPorts(
            FakeTranscriber(
                TranscriptionResult(language="ru", segments=(RecognizedSegment(start=0.0, end=1.0, text="Реплика."),))
            ),
            FakeAligner(),
            FakeDiarizer(),
            FakeSpeakerAssigner(),
        ),
        registry=registry,
        runtime=settings.runtime,
        groq=settings.groq,
        clock=lambda: datetime(2026, 7, 15, tzinfo=UTC),
    )

    with pytest.raises(RoleResolutionRequiredError) as error:
        engine.transcribe(SpeechFile(filename="call.wav", content_type="audio/wav", content=b"RIFF"))

    assert error.value.candidates[0].original_speaker_id == "SPEAKER_00"
    assert error.value.candidates[0].speaker_confidence == 0.8


def test_engine_rejects_no_speech_before_alignment(tmp_path: Path) -> None:
    registry, settings = make_registry(tmp_path)
    transcriber = FakeTranscriber(TranscriptionResult(language="ru", segments=()))
    aligner = FakeAligner()
    engine = CanonicalBatchEngine(
        normalizer=FakeNormalizer(),
        ports=SpeechPorts(transcriber, aligner, FakeDiarizer(), FakeSpeakerAssigner()),
        registry=registry,
        runtime=settings.runtime,
        groq=settings.groq,
    )

    with pytest.raises(NoSpeechError):
        engine.transcribe(SpeechFile(filename="call.wav", content_type="audio/wav", content=b"RIFF"))

    assert transcriber.calls == 1
    assert aligner.calls == 0
