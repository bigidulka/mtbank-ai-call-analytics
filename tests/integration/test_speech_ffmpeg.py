from __future__ import annotations

from pathlib import Path

import pytest

from mtbank_ai.speech.contracts import SpeechFile
from services.speech.media import MediaLimits, MediaNormalizer

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "test_data" / "transport"


@pytest.mark.integration
def test_ffmpeg_normalizes_real_wav_mp3_and_ogg_transport_fixtures_deterministically(tmp_path: Path) -> None:
    normalizer = MediaNormalizer(
        MediaLimits(
            max_upload_bytes=1024 * 1024,
            max_duration_seconds=5.0,
            process_timeout_seconds=10.0,
            temp_root=tmp_path / "work",
        )
    )
    sources = (
        ("silence-16k.wav", "audio/wav"),
        ("silence-16k.mp3", "audio/mpeg"),
        ("silence-16k.ogg", "audio/ogg"),
    )

    normalized_hashes: list[str] = []
    for name, content_type in sources:
        source = SpeechFile(filename=name, content_type=content_type, content=(FIXTURES / name).read_bytes())
        with normalizer.normalize(source) as normalized:
            assert normalized.path.read_bytes()[8:12] == b"WAVE"
            assert normalized.duration_seconds == pytest.approx(1.0, abs=0.03)
            normalized_hashes.append(normalized.audio_sha256)

    repeat_source = SpeechFile(
        filename="silence-16k.mp3",
        content_type="audio/mpeg",
        content=(FIXTURES / "silence-16k.mp3").read_bytes(),
    )
    with normalizer.normalize(repeat_source) as normalized:
        assert normalized.audio_sha256 == normalized_hashes[1]
