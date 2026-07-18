"""Validation of reproducible speech-evaluation manifests and local media evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCORING_METRICS = frozenset({"wer", "der", "role_accuracy", "speaker_attributed_wer"})
SUPPORTED_FORMATS = frozenset({"wav", "mp3", "ogg"})
PUBLIC_ROLES = frozenset({"Оператор", "Клиент"})
_DURATION_TOLERANCE_SECONDS = 0.05
_MEDIA_PROBE_TIMEOUT_SECONDS = 10.0


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestEntry:
    identifier: str
    kind: str
    root: Path
    path: Path
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class _AudioProbe:
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    format: str


@dataclass(frozen=True)
class _ReferenceEvidence:
    roles: frozenset[str]


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[ManifestEntry, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError("manifest.yaml должен быть JSON-совместимым YAML") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ManifestError("schema_version=1 обязателен")
    dataset = payload.get("dataset")
    entries = payload.get("entries")
    if not isinstance(dataset, dict) or not isinstance(entries, list):
        raise ManifestError("dataset и entries обязательны")

    seen_ids: set[str] = set()
    parsed: list[ManifestEntry] = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise ManifestError("entry должен быть object")
        entry = _parse_entry(path.parent, raw)
        if entry.identifier in seen_ids:
            raise ManifestError("entry.id должен быть уникальным")
        seen_ids.add(entry.identifier)
        parsed.append(entry)
    if not parsed:
        raise ManifestError("manifest должен содержать хотя бы один fixture")
    return payload, tuple(parsed)


def validate_manifest(path: Path, *, require_release_corpus: bool) -> tuple[ManifestEntry, ...]:
    payload, entries = load_manifest(path)
    status = payload["dataset"].get("status")
    if status not in {"transport_only_release_blocked", "release_ready"}:
        raise ManifestError("dataset.status не поддерживается")
    if require_release_corpus:
        if status != "release_ready":
            raise ManifestError("release gate: dataset.status должен быть release_ready")
        _validate_unique_release_audio(entries)

    references: dict[str, _ReferenceEvidence] = {}
    for entry in entries:
        _validate_file(entry)
        _validate_audio_properties(entry)
        reference = _validate_provenance(entry)
        if reference is not None:
            references[entry.identifier] = reference
    if require_release_corpus:
        _validate_release_corpus(entries, references)
    return entries


def _parse_entry(root: Path, raw: dict[str, Any]) -> ManifestEntry:
    base_fields = {
        "id",
        "kind",
        "path",
        "sha256",
        "format",
        "sample_rate_hz",
        "channels",
        "duration_seconds",
        "license",
        "provenance",
        "eligible_for",
        "excluded_from",
    }
    reference_fields = {"reference_path", "reference_sha256", "speaker_count"}
    identifier = raw.get("id")
    kind = raw.get("kind")
    if not isinstance(identifier, str) or not identifier or kind not in {"transport_only", "speech_reference"}:
        raise ManifestError("entry id/kind некорректны")
    required_fields = base_fields if kind == "transport_only" else base_fields | reference_fields
    if set(raw) != required_fields:
        raise ManifestError("entry содержит неизвестные или отсутствующие поля")

    fixture_path = _resolve_dataset_path(root.resolve(), raw["path"], "entry.path")
    duration = raw["duration_seconds"]
    sample_rate = raw["sample_rate_hz"]
    channels = raw["channels"]
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ManifestError("duration_seconds должен быть положительным числом")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ManifestError("sample_rate_hz должен быть положительным integer")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
        raise ManifestError("channels должен быть положительным integer")
    if raw["format"] not in SUPPORTED_FORMATS:
        raise ManifestError("format не поддерживается")
    return ManifestEntry(
        identifier=identifier,
        kind=kind,
        root=root.resolve(),
        path=fixture_path,
        duration_seconds=float(duration),
        sample_rate_hz=sample_rate,
        channels=channels,
        raw=raw,
    )


def _resolve_dataset_path(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} некорректен")
    target = (root / value).resolve()
    if target == root or root not in target.parents:
        raise ManifestError(f"{field} выходит за test_data")
    return target


def _validate_unique_release_audio(entries: tuple[ManifestEntry, ...]) -> None:
    paths = tuple(entry.path for entry in entries)
    hashes = tuple(entry.raw["sha256"] for entry in entries)
    if len(set(paths)) != len(paths):
        raise ManifestError("release gate: audio paths должны быть уникальны")
    if len(set(hashes)) != len(hashes):
        raise ManifestError("release gate: audio SHA-256 должны быть уникальны")


def _validate_file(entry: ManifestEntry) -> None:
    expected_hash = _require_sha256(entry.raw["sha256"], "sha256")
    if not entry.path.is_file() or entry.path.is_symlink():
        raise ManifestError(f"fixture отсутствует: {entry.identifier}")
    if _sha256(entry.path) != expected_hash:
        raise ManifestError(f"SHA-256 не совпадает: {entry.identifier}")


def _validate_audio_properties(entry: ManifestEntry) -> None:
    probe = _probe_audio(entry.path, entry.raw["format"])
    if probe.format != entry.raw["format"]:
        raise ManifestError(f"format не совпадает с фактическим аудио: {entry.identifier}")
    if probe.sample_rate_hz != entry.sample_rate_hz:
        raise ManifestError(f"sample_rate_hz не совпадает с фактическим аудио: {entry.identifier}")
    if probe.channels != entry.channels:
        raise ManifestError(f"channels не совпадает с фактическим аудио: {entry.identifier}")
    if not math.isclose(
        probe.duration_seconds,
        entry.duration_seconds,
        rel_tol=0.0,
        abs_tol=_duration_tolerance(entry.duration_seconds),
    ):
        raise ManifestError(f"duration_seconds не совпадает с фактическим аудио: {entry.identifier}")


def _probe_audio(path: Path, declared_format: object) -> _AudioProbe:
    if declared_format == "wav":
        return _probe_wav(path)
    return _probe_with_ffprobe(path)


def _probe_wav(path: Path) -> _AudioProbe:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            sample_width = audio.getsampwidth()
        if channels <= 0 or sample_rate <= 0 or frames <= 0 or sample_width <= 0:
            raise ValueError("invalid WAV metadata")
        if frames * channels * sample_width > path.stat().st_size:
            raise ValueError("truncated WAV data")
    except (OSError, EOFError, ValueError, wave.Error) as error:
        raise ManifestError(f"не удалось probe WAV fixture: {path.name}") from error
    return _AudioProbe(
        duration_seconds=frames / sample_rate,
        sample_rate_hz=sample_rate,
        channels=channels,
        format="wav",
    )


def _probe_with_ffprobe(path: Path) -> _AudioProbe:
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-show_entries",
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
    )
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_MEDIA_PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise ValueError("ffprobe failed")
        payload = json.loads(result.stdout)
        streams = payload.get("streams") if isinstance(payload, dict) else None
        format_payload = payload.get("format") if isinstance(payload, dict) else None
        if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
            raise ValueError("missing audio stream")
        if not isinstance(format_payload, dict):
            raise ValueError("missing media format")
        sample_rate = int(streams[0]["sample_rate"])
        channels = streams[0]["channels"]
        duration = float(format_payload["duration"])
        format_names = format_payload["format_name"]
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("invalid channels")
        if sample_rate <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid audio metadata")
        if not isinstance(format_names, str):
            raise ValueError("invalid media format")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        raise ManifestError(f"не удалось ffprobe fixture: {path.name}") from error

    known_formats = set(format_names.split(","))
    actual_format = next((value for value in SUPPORTED_FORMATS if value in known_formats), "")
    return _AudioProbe(
        duration_seconds=duration,
        sample_rate_hz=sample_rate,
        channels=channels,
        format=actual_format,
    )


def _duration_tolerance(duration_seconds: float) -> float:
    return max(_DURATION_TOLERANCE_SECONDS, duration_seconds * 0.001)


def _validate_provenance(entry: ManifestEntry) -> _ReferenceEvidence | None:
    raw = entry.raw
    if not isinstance(raw["license"], str) or not raw["license"].strip():
        raise ManifestError("license обязателен")
    if not isinstance(raw["provenance"], str) or not raw["provenance"].strip():
        raise ManifestError("provenance обязателен")
    eligible = raw["eligible_for"]
    excluded = raw["excluded_from"]
    if (
        not isinstance(eligible, list)
        or not isinstance(excluded, list)
        or not all(isinstance(item, str) for item in (*eligible, *excluded))
    ):
        raise ManifestError("eligible_for/excluded_from должны быть списками строк")
    if len(set(eligible)) != len(eligible) or len(set(excluded)) != len(excluded) or set(eligible) & set(excluded):
        raise ManifestError("eligible_for/excluded_from не должны пересекаться или дублироваться")
    if entry.kind == "transport_only":
        if SCORING_METRICS & set(eligible) or not SCORING_METRICS.issubset(set(excluded)):
            raise ManifestError("transport_only fixture запрещён для speech scoring")
        return None

    reference_path = _resolve_dataset_path(entry.root, raw["reference_path"], "reference_path")
    expected_reference_hash = _require_sha256(raw["reference_sha256"], "reference_sha256")
    if not reference_path.is_file() or reference_path.is_symlink():
        raise ManifestError("speech_reference требует существующий reference_path")
    if _sha256(reference_path) != expected_reference_hash:
        raise ManifestError("reference SHA-256 не совпадает")
    reference = _load_reference(reference_path, entry.duration_seconds)
    speaker_count = raw["speaker_count"]
    if not isinstance(speaker_count, int) or isinstance(speaker_count, bool) or speaker_count < 1:
        raise ManifestError("speech_reference требует speaker_count")
    if speaker_count != len(reference.roles):
        raise ManifestError("speaker_count должен совпадать с role labels в reference")
    if not SCORING_METRICS.issubset(set(eligible)) or SCORING_METRICS & set(excluded):
        raise ManifestError("speech_reference должен быть eligible для всех speech metrics")
    return reference


def _load_reference(path: Path, audio_duration_seconds: float) -> _ReferenceEvidence:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"reference не читается: {path.name}") from error
    if not isinstance(payload, dict) or set(payload) != {"segments"}:
        raise ManifestError("reference требует точную schema с segments")
    segments = payload["segments"]
    if not isinstance(segments, list) or not segments:
        raise ManifestError("reference требует непустой segments list")

    identifiers: set[str] = set()
    roles: set[str] = set()
    previous_start = -1.0
    for raw_segment in segments:
        if not isinstance(raw_segment, dict) or set(raw_segment) != {"id", "start", "end", "speaker", "text"}:
            raise ManifestError("reference segment требует transcript, role label и timestamps")
        identifier = raw_segment["id"]
        speaker = raw_segment["speaker"]
        text = raw_segment["text"]
        start = raw_segment["start"]
        end = raw_segment["end"]
        if not all(isinstance(value, str) and value.strip() for value in (identifier, speaker, text)):
            raise ManifestError("reference segment id/speaker/text должны быть непустыми строками")
        if speaker not in PUBLIC_ROLES:
            raise ManifestError("reference role labels должны быть Оператор или Клиент")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or start >= end
            or end > audio_duration_seconds + _duration_tolerance(audio_duration_seconds)
        ):
            raise ManifestError("reference timestamps некорректны или выходят за audio duration")
        if start < previous_start:
            raise ManifestError("reference segments должны быть отсортированы по start")
        if identifier in identifiers:
            raise ManifestError("reference segment IDs должны быть уникальны")
        identifiers.add(identifier)
        roles.add(speaker)
        previous_start = float(start)
    return _ReferenceEvidence(roles=frozenset(roles))


def _validate_release_corpus(
    entries: tuple[ManifestEntry, ...],
    references: dict[str, _ReferenceEvidence],
) -> None:
    speech = tuple(entry for entry in entries if entry.kind == "speech_reference")
    if len(speech) < 5:
        raise ManifestError("release gate: требуется минимум 5 licensed speech_reference fixtures")
    if sum(entry.duration_seconds for entry in speech) < 300.0:
        raise ManifestError("release gate: требуется минимум 5 минут речи")
    if not any(entry.sample_rate_hz == 8000 for entry in speech):
        raise ManifestError("release gate: требуется минимум один 8 kHz fixture")
    if not any(len(references[entry.identifier].roles) >= 2 and entry.duration_seconds >= 60.0 for entry in speech):
        raise ManifestError("release gate: требуется диалог двух speaker длительностью не менее минуты")


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not all(char in "0123456789abcdef" for char in value):
        raise ManifestError(f"{field} должен быть lower-case SHA-256")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
