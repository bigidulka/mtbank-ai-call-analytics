"""Groq batch ASR and local offline pyannote Community-1 adapters."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from mtbank_ai.domain.transcript import ASRProviderMetadata, WordTimestamp
from mtbank_ai.speech.contracts import (
    AlignedSegment,
    DiarizationTurn,
    RecognizedSegment,
    RecognizedWord,
    SpeakerAttributedSegment,
    TranscriptionResult,
)
from services.speech.errors import SpeechProviderError
from services.speech.manifest import ModelRegistry
from services.speech.media import NormalizedAudio
from services.speech.ports import AlignerPort, DiarizerPort, SpeakerAssignerPort, SpeechPorts, TranscriberPort
from services.speech.settings import GroqTranscriptionSettings, SpeechRuntimeSettings


class GroqWhisperTranscriber(TranscriberPort):
    """Makes exactly one bounded OpenAI-compatible Groq ASR request per batch."""

    def __init__(
        self,
        settings: GroqTranscriptionSettings,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory

    def transcribe(self, audio: NormalizedAudio, *, language: str) -> TranscriptionResult:
        if language != self._settings.language:
            raise SpeechProviderError("configured Groq language does not match canonical language")
        try:
            with audio.path.open("rb") as source, self._client_factory(
                timeout=httpx.Timeout(
                    self._settings.request_timeout_seconds,
                    connect=self._settings.connect_timeout_seconds,
                ),
                trust_env=False,
                follow_redirects=False,
            ) as client:
                with client.stream(
                    "POST",
                    str(self._settings.endpoint),
                    files=(
                        ("file", (audio.path.name, source, "audio/wav")),
                        ("model", (None, self._settings.model)),
                        ("language", (None, language)),
                        ("temperature", (None, str(self._settings.temperature))),
                        ("response_format", (None, self._settings.response_format)),
                        ("timestamp_granularities[]", (None, "segment")),
                        ("timestamp_granularities[]", (None, "word")),
                    ),
                    headers={
                        "Authorization": f"Bearer {self._settings.api_key.get_secret_value()}",
                        "Accept-Encoding": "identity",
                    },
                ) as response:
                    content = _read_bounded_response(response, self._settings.max_response_bytes)
                    if not 200 <= response.status_code < 300:
                        raise SpeechProviderError("Groq transcription request failed")
                    payload = json.loads(content)
                    return _groq_result(
                        payload,
                        language=language,
                        endpoint_fingerprint=self._settings.endpoint_fingerprint,
                        request_id=response.headers.get("x-request-id"),
                        model=self._settings.model,
                    )
        except SpeechProviderError:
            raise
        except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
            raise SpeechProviderError("Groq transcription request failed") from error


class PassthroughTimestampAligner(AlignerPort):
    """Groq verbose JSON timestamps are already aligned; no local alignment model exists."""

    def align(
        self,
        audio: NormalizedAudio,
        transcription: TranscriptionResult,
        *,
        language: str,
    ) -> tuple[AlignedSegment, ...]:
        del audio
        if language != transcription.language:
            raise SpeechProviderError("Groq transcription language does not match canonical language")
        return tuple(
            AlignedSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=tuple(
                    WordTimestamp(
                        word=word.text,
                        start=word.start,
                        end=word.end,
                        confidence=word.confidence,
                    )
                    for word in segment.words
                ),
            )
            for segment in transcription.segments
        )


class PyannoteCommunityDiarizer(DiarizerPort):
    """Loads only a pre-fetched local Community-1 artifact; no runtime token/download."""

    def __init__(self, *, model_path: Path, device: str) -> None:
        self._model_path = model_path
        self._device = device
        self._pipeline: Any | None = None

    def diarize(self, audio: NormalizedAudio) -> tuple[DiarizationTurn, ...]:
        try:
            result = self._get_pipeline()(str(audio.path))
            annotation = getattr(result, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = getattr(result, "speaker_diarization", result)
            turns = tuple(
                DiarizationTurn(
                    original_speaker_id=str(speaker),
                    start=float(turn.start),
                    end=float(turn.end),
                )
                for turn, _, speaker in annotation.itertracks(yield_label=True)
                if _valid_interval(float(turn.start), float(turn.end))
            )
            return tuple(sorted(turns, key=lambda item: (item.start, item.end, item.original_speaker_id)))
        except SpeechProviderError:
            raise
        except Exception as error:
            raise SpeechProviderError("pyannote diarization failed") from error

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                import torch  # pyright: ignore[reportMissingImports]
                from pyannote.audio import Pipeline  # pyright: ignore[reportMissingImports]

                pipeline = Pipeline.from_pretrained(str(self._model_path), token=None, cache_dir=str(self._model_path))
                if pipeline is None:
                    raise RuntimeError("Pipeline.from_pretrained returned None")
                pipeline.to(torch.device(self._device))
                self._pipeline = pipeline
            except Exception as error:
                raise SpeechProviderError("pyannote Community-1 model load failed") from error
        return self._pipeline


@dataclass(frozen=True, slots=True)
class _SpeakerWord:
    word: WordTimestamp
    speaker_id: str | None
    overlap: float


class LocalOverlapSpeakerAssigner(SpeakerAssignerPort):
    """Deterministically assigns each Groq word to the maximum-overlap pyannote turn."""

    def assign(
        self,
        aligned_segments: tuple[AlignedSegment, ...],
        diarization: tuple[DiarizationTurn, ...],
    ) -> tuple[SpeakerAttributedSegment, ...]:
        words = tuple(word for segment in aligned_segments for word in segment.words)
        if not words:
            raise SpeechProviderError("Groq response has no word timestamps")
        attributed = tuple(_SpeakerWord(word, *_best_speaker(word, diarization)) for word in words)
        result: list[SpeakerAttributedSegment] = []
        group: list[_SpeakerWord] = []
        for item in attributed:
            if group and item.speaker_id != group[-1].speaker_id:
                result.append(_speaker_segment(group))
                group = []
            group.append(item)
        if group:
            result.append(_speaker_segment(group))
        return tuple(result)


def build_production_ports(
    registry: ModelRegistry,
    runtime: SpeechRuntimeSettings,
    groq: GroqTranscriptionSettings,
) -> SpeechPorts:
    """Constructs the one Groq/pyannote adapter graph after local artifact verification."""

    return SpeechPorts(
        transcriber=GroqWhisperTranscriber(groq),
        aligner=PassthroughTimestampAligner(),
        diarizer=PyannoteCommunityDiarizer(
            model_path=registry.artifact_path(registry.manifest.diarization),
            device=runtime.device,
        ),
        speaker_assigner=LocalOverlapSpeakerAssigner(),
    )


def _read_bounded_response(response: httpx.Response, maximum_bytes: int) -> bytes:
    if response.headers.get("content-encoding", "identity").strip().casefold() != "identity":
        raise SpeechProviderError("Groq transcription response uses unsupported content encoding")
    content_length = response.headers.get("content-length")
    if content_length is not None and (not content_length.isdecimal() or int(content_length) > maximum_bytes):
        raise SpeechProviderError("Groq transcription response exceeded bound")
    content = bytearray()
    for chunk in response.iter_raw():
        if len(chunk) > maximum_bytes - len(content):
            raise SpeechProviderError("Groq transcription response exceeded bound")
        content.extend(chunk)
    return bytes(content)


def _groq_result(
    payload: object,
    *,
    language: str,
    endpoint_fingerprint: str,
    request_id: str | None,
    model: str,
) -> TranscriptionResult:
    if not isinstance(payload, dict):
        raise SpeechProviderError("Groq transcription response is invalid")
    raw_segments = payload.get("segments")
    raw_words = payload.get("words")
    if not isinstance(raw_segments, list) or not isinstance(raw_words, list):
        raise SpeechProviderError("Groq response lacks verbose timestamps")
    words = tuple(_recognized_word(item) for item in raw_words)
    segments = tuple(_recognized_segment(item, words) for item in raw_segments)
    if not segments:
        return TranscriptionResult(language=language, segments=())
    usage_seconds = _usage_seconds(payload)
    try:
        metadata = ASRProviderMetadata(
            provider="groq",
            model=model,
            endpoint_fingerprint=endpoint_fingerprint,
            request_id=request_id.strip() if isinstance(request_id, str) and request_id.strip() else None,
            usage_seconds=usage_seconds,
        )
        return TranscriptionResult(language=language, segments=segments, provider_metadata=metadata)
    except ValueError as error:
        raise SpeechProviderError("Groq transcription response is invalid") from error


def _recognized_segment(raw: object, words: tuple[RecognizedWord, ...]) -> RecognizedSegment:
    if not isinstance(raw, dict):
        raise SpeechProviderError("Groq segment is invalid")
    try:
        start = _finite_float(raw["start"])
        end = _finite_float(raw["end"])
        text = str(raw["text"]).strip()
        segment_words = tuple(
            word
            for word in words
            if _overlap(start, end, word.start, word.end) > 0.0
            and word.start >= start
            and word.end <= end
        )
        return RecognizedSegment(start=start, end=end, text=text, words=segment_words)
    except (KeyError, ValueError) as error:
        raise SpeechProviderError("Groq segment is invalid") from error


def _recognized_word(raw: object) -> RecognizedWord:
    if not isinstance(raw, dict):
        raise SpeechProviderError("Groq word is invalid")
    try:
        confidence = raw.get("probability")
        return RecognizedWord(
            text=str(raw.get("word", "")).strip(),
            start=_finite_float(raw["start"]),
            end=_finite_float(raw["end"]),
            confidence=_optional_confidence(confidence),
        )
    except (KeyError, ValueError) as error:
        raise SpeechProviderError("Groq word is invalid") from error


def _usage_seconds(payload: dict[str, object]) -> float | None:
    usage = payload.get("usage")
    candidate = usage.get("seconds") if isinstance(usage, dict) else payload.get("duration")
    if candidate is None:
        return None
    value = _finite_float(candidate)
    return value if value >= 0.0 else None


def _best_speaker(word: WordTimestamp, turns: tuple[DiarizationTurn, ...]) -> tuple[str | None, float]:
    candidates = tuple(
        (
            _overlap(word.start, word.end, turn.start, turn.end),
            turn.start,
            turn.end,
            turn.original_speaker_id,
        )
        for turn in turns
    )
    if not candidates:
        return None, 0.0
    overlap, _, _, speaker_id = sorted(candidates, key=lambda item: (-round(item[0], 9), item[1], item[2], item[3]))[0]
    return (speaker_id, overlap) if overlap > 0.0 else (None, 0.0)


def _speaker_segment(group: list[_SpeakerWord]) -> SpeakerAttributedSegment:
    words = tuple(item.word for item in group)
    duration = sum(word.end - word.start for word in words)
    overlap = sum(item.overlap for item in group)
    return SpeakerAttributedSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.word for word in words),
        words=words,
        original_speaker_id=group[0].speaker_id,
        speaker_confidence=overlap / duration if duration > 0.0 and group[0].speaker_id is not None else None,
    )


def _finite_float(value: object) -> float:
    if not isinstance(value, (float, int, str)):
        raise ValueError("timestamp is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("timestamp is not finite")
    return result


def _optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    result = _finite_float(value)
    return result if 0.0 <= result <= 1.0 else None


def _overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _valid_interval(start: float, end: float) -> bool:
    return math.isfinite(start) and math.isfinite(end) and start >= 0.0 and start < end
