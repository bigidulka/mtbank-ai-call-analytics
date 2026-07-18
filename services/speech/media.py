"""Safe local media admission and deterministic FFmpeg normalization."""

from __future__ import annotations

import hashlib
import math
import os
import signal
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mtbank_ai.speech.contracts import SpeechFile
from services.speech.errors import MediaTimeoutError, MediaValidationError, UnsupportedMediaError

_MAX_PROCESS_OUTPUT_BYTES = 64 * 1024
_NORMALIZED_BYTES_PER_SAMPLE = 2


@dataclass(frozen=True, slots=True)
class MediaLimits:
    max_upload_bytes: int
    max_duration_seconds: float
    process_timeout_seconds: float
    temp_root: Path
    sample_rate_hz: int = 16_000
    channels: int = 1
    codec: str = "pcm_s16le"


@dataclass(frozen=True, slots=True)
class NormalizedAudio:
    path: Path
    duration_seconds: float
    audio_sha256: str
    source_format: str


class MediaNormalizer:
    """Validates magic/MIME and returns a temporary deterministic mono WAV."""

    def __init__(self, limits: MediaLimits) -> None:
        self._limits = limits

    @contextmanager
    def normalize(self, source: SpeechFile) -> Generator[NormalizedAudio, None, None]:
        source_format = self._validate_source(source)
        temp_root = self._prepare_temp_root()
        with tempfile.TemporaryDirectory(prefix="speech-", dir=temp_root) as workspace_text:
            workspace = Path(workspace_text)
            source_path = workspace / f"source.{source_format}"
            _write_private_file(source_path, source.content)
            duration_seconds = self._probe_duration(source_path, workspace, "source")
            if duration_seconds > self._limits.max_duration_seconds:
                raise MediaValidationError("audio duration exceeds configured limit")

            normalized_path = workspace / "normalized.wav"
            self._normalize(source_path, normalized_path, workspace)
            normalized_duration = self._probe_duration(normalized_path, workspace, "normalized")
            if normalized_duration > self._limits.max_duration_seconds:
                raise MediaValidationError("normalized audio duration exceeds configured limit")
            if normalized_path.stat().st_size > self._max_normalized_bytes():
                raise MediaValidationError("normalized audio exceeds configured limit")
            yield NormalizedAudio(
                path=normalized_path,
                duration_seconds=normalized_duration,
                audio_sha256=_sha256_file(normalized_path),
                source_format=source_format,
            )

    def _validate_source(self, source: SpeechFile) -> str:
        if not source.content:
            raise MediaValidationError("empty audio")
        if len(source.content) > self._limits.max_upload_bytes:
            raise MediaValidationError("upload exceeds configured limit")
        detected_format = _detect_format(source.content)
        declared_format = _format_for_mime(source.content_type)
        if detected_format is None:
            raise MediaValidationError("unrecognised audio magic")
        if declared_format is None or declared_format != detected_format:
            raise UnsupportedMediaError("declared MIME does not match audio magic")
        return detected_format

    def _prepare_temp_root(self) -> Path:
        self._limits.temp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._limits.temp_root.is_symlink() or not self._limits.temp_root.is_dir():
            raise MediaValidationError("temporary workspace is unavailable")
        return self._limits.temp_root.resolve()

    def _probe_duration(self, input_path: Path, workspace: Path, label: str) -> float:
        output = self._run(
            (
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(input_path),
            ),
            workspace,
            label,
            capture_stdout=True,
        )
        try:
            duration = float(output.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as error:
            raise MediaValidationError("FFprobe did not return an audio duration") from error
        if not math.isfinite(duration) or duration <= 0:
            raise MediaValidationError("audio duration is not finite and positive")
        return duration

    def _normalize(self, source_path: Path, normalized_path: Path, workspace: Path) -> None:
        self._run(
            (
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-i",
                str(source_path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-ac",
                str(self._limits.channels),
                "-ar",
                str(self._limits.sample_rate_hz),
                "-c:a",
                self._limits.codec,
                "-fflags",
                "+bitexact",
                "-flags:a",
                "+bitexact",
                "-y",
                str(normalized_path),
            ),
            workspace,
            "ffmpeg",
            capture_stdout=False,
        )
        if not normalized_path.is_file() or normalized_path.is_symlink() or normalized_path.stat().st_size == 0:
            raise MediaValidationError("FFmpeg did not produce a regular normalized WAV")

    def _run(
        self,
        argv: tuple[str, ...],
        workspace: Path,
        label: str,
        *,
        capture_stdout: bool,
    ) -> bytes:
        stdout_path = workspace / f".{label}.stdout"
        stderr_path = workspace / f".{label}.stderr"
        with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file if capture_stdout else subprocess.DEVNULL,
                    stderr=stderr_file,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError as error:
                raise MediaValidationError("FFmpeg tooling is unavailable") from error
            try:
                return_code = process.wait(timeout=self._limits.process_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                _terminate_process_group(process)
                raise MediaTimeoutError("media process exceeded timeout") from error
        output_too_large = (
            stdout_path.stat().st_size > _MAX_PROCESS_OUTPUT_BYTES
            or stderr_path.stat().st_size > _MAX_PROCESS_OUTPUT_BYTES
        )
        if output_too_large:
            raise MediaValidationError("media tool output exceeded configured limit")
        if return_code != 0:
            raise MediaValidationError("media tool rejected audio")
        return stdout_path.read_bytes() if capture_stdout else b""

    def _max_normalized_bytes(self) -> int:
        samples = int(math.ceil(self._limits.max_duration_seconds * self._limits.sample_rate_hz))
        return samples * self._limits.channels * _NORMALIZED_BYTES_PER_SAMPLE + 4096


def _detect_format(content: bytes) -> str | None:
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return "wav"
    if content.startswith(b"OggS"):
        return "ogg"
    if content.startswith(b"ID3") or (len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0):
        return "mp3"
    return None


def _format_for_mime(content_type: str) -> str | None:
    normalized = content_type.partition(";")[0].strip().casefold()
    if normalized in {"audio/wav", "audio/x-wav"}:
        return "wav"
    if normalized in {"audio/mpeg", "audio/mp3"}:
        return "mp3"
    if normalized in {"audio/ogg", "application/ogg"}:
        return "ogg"
    return None


def _write_private_file(path: Path, content: bytes) -> None:
    with path.open("xb") as output:
        output.write(content)
    os.chmod(path, 0o600)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as media_file:
        while chunk := media_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        process.wait()
