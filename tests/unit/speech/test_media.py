from __future__ import annotations

from pathlib import Path

import pytest

from mtbank_ai.speech.contracts import SpeechFile
from services.speech.errors import MediaTimeoutError, MediaValidationError, UnsupportedMediaError
from services.speech.media import MediaLimits, MediaNormalizer

ROOT = Path(__file__).parents[3]
FIXTURES = ROOT / "test_data" / "transport"


def _normalizer(tmp_path: Path, *, max_duration_seconds: float = 5.0) -> MediaNormalizer:
    return MediaNormalizer(
        MediaLimits(
            max_upload_bytes=1024 * 1024,
            max_duration_seconds=max_duration_seconds,
            process_timeout_seconds=5.0,
            temp_root=tmp_path / "work",
        )
    )


def _source(name: str, content_type: str) -> SpeechFile:
    return SpeechFile(filename=name, content_type=content_type, content=(FIXTURES / name).read_bytes())


def test_media_normalizer_validates_magic_and_cleans_private_workspace(tmp_path: Path) -> None:
    normalizer = _normalizer(tmp_path)
    workspace_root = tmp_path / "work"

    with normalizer.normalize(_source("silence-16k.wav", "audio/wav")) as normalized:
        normalized_path = normalized.path
        assert normalized_path.is_file()
        assert normalized_path.read_bytes()[:12] == b"RIFF" + normalized_path.read_bytes()[4:8] + b"WAVE"
        assert normalized.duration_seconds == pytest.approx(1.0, abs=0.01)
        assert len(normalized.audio_sha256) == 64

    assert not normalized_path.exists()
    assert workspace_root.exists()
    assert list(workspace_root.iterdir()) == []


def test_media_normalizer_rejects_corrupt_magic_and_declared_mime_mismatch(tmp_path: Path) -> None:
    normalizer = _normalizer(tmp_path)

    with pytest.raises(MediaValidationError):
        with normalizer.normalize(SpeechFile(filename="bad.wav", content_type="audio/wav", content=b"not audio")):
            pass
    with pytest.raises(UnsupportedMediaError):
        with normalizer.normalize(_source("silence-16k.wav", "audio/mpeg")):
            pass


def test_media_normalizer_rejects_duration_above_limit(tmp_path: Path) -> None:
    with pytest.raises(MediaValidationError):
        with _normalizer(tmp_path, max_duration_seconds=0.5).normalize(_source("silence-16k.ogg", "audio/ogg")):
            pass


class TimeoutNormalizer(MediaNormalizer):
    def _probe_duration(self, input_path: Path, workspace: Path, label: str) -> float:
        del input_path, workspace, label
        raise MediaTimeoutError("test timeout")


def test_media_timeout_still_cleans_temporary_input(tmp_path: Path) -> None:
    normalizer = TimeoutNormalizer(
        MediaLimits(
            max_upload_bytes=1024 * 1024,
            max_duration_seconds=5.0,
            process_timeout_seconds=0.01,
            temp_root=tmp_path / "work",
        )
    )

    with pytest.raises(MediaTimeoutError):
        with normalizer.normalize(_source("silence-16k.wav", "audio/wav")):
            pass

    assert list((tmp_path / "work").iterdir()) == []
