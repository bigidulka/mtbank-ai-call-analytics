#!/usr/bin/env python3
"""Normalize one fixture to the canonical 16 kHz mono PCM WAV transport form."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def normalize(source: Path, destination: Path, *, overwrite: bool) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError("input должен быть обычным существующим файлом")
    if destination.exists() and not overwrite:
        raise ValueError("output уже существует; передайте --overwrite только осознанно")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-y",
            str(destination),
        ),
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    normalize(arguments.input, arguments.output, overwrite=arguments.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
