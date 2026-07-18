#!/usr/bin/env python3
"""Создаёт малые silence-only fixtures для codec/transport тестов, не для ASR eval."""

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

_SAMPLE_RATE = 16_000
_DURATION_SECONDS = 1


def generate(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    wav_path = destination / "silence-16k.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_SAMPLE_RATE)
        output.writeframes(b"\x00\x00" * _SAMPLE_RATE * _DURATION_SECONDS)

    _convert(wav_path, destination / "silence-16k.mp3", ("-c:a", "libmp3lame", "-q:a", "9"))
    _convert(wav_path, destination / "silence-16k.ogg", ("-c:a", "libvorbis", "-q:a", "0"))


def _convert(source: Path, destination: Path, codec_args: tuple[str, ...]) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-map_metadata",
            "-1",
            *codec_args,
            str(destination),
        ),
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
