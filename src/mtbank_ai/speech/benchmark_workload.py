"""Общие helpers exact-five-minute workload для устанавливаемых SLA CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


class BenchmarkWorkloadError(ValueError):
    """Five-minute workload нельзя создать или проверить."""


def make_five_minutes(source: Path, destination: Path) -> None:
    command = (
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stream_loop",
        "-1",
        "-i",
        str(source),
        "-t",
        "300",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(destination),
    )
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkWorkloadError("не удалось создать five-minute synthetic workload") from error


def duration_seconds(path: Path) -> float:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    )
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        duration = float(completed.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise BenchmarkWorkloadError("не удалось определить duration workload") from error
    if not 299.9 <= duration <= 300.1:
        raise BenchmarkWorkloadError("workload должен быть ровно 300 секунд")
    return duration
