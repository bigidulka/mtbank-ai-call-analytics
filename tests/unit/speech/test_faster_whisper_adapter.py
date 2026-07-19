from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from services.speech.adapters import FasterWhisperTranscriber
from services.speech.errors import SpeechProviderError
from services.speech.media import NormalizedAudio
from services.speech.settings import FasterWhisperSettings


@dataclass(frozen=True)
class _Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class _Segment:
    start: float
    end: float
    text: str
    words: tuple[_Word, ...]


class _Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, path: str, **kwargs: object) -> tuple[tuple[_Segment, ...], object]:
        self.calls.append({"path": path, **kwargs})
        return (
            (
                _Segment(
                    start=0.0,
                    end=1.0,
                    text="Добрый день",
                    words=(
                        _Word("Добрый", 0.0, 0.4, 0.9),
                        _Word("день", 0.4, 1.0, 0.8),
                    ),
                ),
            ),
            object(),
        )


def _audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "normalized.wav"
    path.write_bytes(b"RIFF")
    return NormalizedAudio(path=path, duration_seconds=1.0, audio_sha256="a" * 64, source_format="wav")


def test_local_faster_whisper_uses_verified_path_int8_and_native_word_timestamps(tmp_path: Path) -> None:
    artifact = tmp_path / "asr"
    artifact.mkdir()
    model = _Model()
    loaded: dict[str, object] = {}

    def factory(path: str, **kwargs: object) -> _Model:
        loaded.update(path=path, **kwargs)
        return model

    result = FasterWhisperTranscriber(
        model_path=artifact,
        settings=FasterWhisperSettings(),
        device="cpu",
        model_factory=factory,
    ).transcribe(_audio(tmp_path), language="ru")

    assert loaded == {"path": str(artifact), "device": "cpu", "compute_type": "int8", "cpu_threads": 8}
    assert model.calls == [
        {
            "path": str(tmp_path / "normalized.wav"),
            "language": "ru",
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": False,
            "word_timestamps": True,
            "condition_on_previous_text": True,
        }
    ]
    assert result.provider_metadata is None
    assert result.segments[0].words[0].text == "Добрый"


def test_local_faster_whisper_fails_closed_for_missing_artifact_or_word_timestamps(tmp_path: Path) -> None:
    transcriber = FasterWhisperTranscriber(
        model_path=tmp_path / "missing",
        settings=FasterWhisperSettings(),
        device="cpu",
    )

    with pytest.raises(SpeechProviderError, match="artifact"):
        transcriber.transcribe(_audio(tmp_path), language="ru")

    artifact = tmp_path / "asr"
    artifact.mkdir()

    class ModelWithoutWords:
        def transcribe(self, path: str, **kwargs: object) -> tuple[tuple[_Segment, ...], object]:
            del path, kwargs
            return ((_Segment(0.0, 1.0, "Текст", ()),), object())

    def factory(*args: object, **kwargs: object) -> ModelWithoutWords:
        del args, kwargs
        return ModelWithoutWords()

    with pytest.raises(SpeechProviderError, match="segment"):
        FasterWhisperTranscriber(
            model_path=artifact,
            settings=FasterWhisperSettings(),
            device="cpu",
            model_factory=factory,
        ).transcribe(_audio(tmp_path), language="ru")
