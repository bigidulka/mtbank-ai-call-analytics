#!/usr/bin/env python3
"""Последовательно оценивает canonical Groq+Community-1 speech service без сохранения текста."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

if __package__:
    from .evaluate_speech import (
        ErrorCounts,
        Segment,
        _counts_json,
        corpus_wer,
        diarization_error_rate,
        load_segments,
        speaker_attributed_wer,
        time_weighted_role_accuracy,
    )
else:
    from evaluate_speech import (
        ErrorCounts,
        Segment,
        _counts_json,
        corpus_wer,
        diarization_error_rate,
        load_segments,
        speaker_attributed_wer,
        time_weighted_role_accuracy,
    )
from mtbank_ai.speech.contracts import SpeechTranscriptionResponse
from mtbank_ai.speech.dataset import ManifestEntry, validate_manifest

_CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}


class CanonicalEvaluationFailure(RuntimeError):
    def __init__(self, *, status_code: int, reason: str) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("canonical base URL должен быть абсолютным HTTP(S) URL без query/fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("canonical base URL не должен содержать path")
    return f"{base_url.rstrip('/')}/v1/transcribe"


def _segments(response: SpeechTranscriptionResponse) -> tuple[Segment, ...]:
    return tuple(
        Segment(
            identifier=str(segment.id),
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker.value,
            text=segment.text,
        )
        for segment in response.transcript.segments
    )


def _component_revisions(response: SpeechTranscriptionResponse) -> dict[str, dict[str, str | None]]:
    metadata = response.transcript.asr_metadata
    return {
        name: {
            "package": component.package,
            "package_version": component.package_version,
            "model_id": component.model_id,
            "model_revision": component.model_revision,
            "artifact_sha256": component.artifact_sha256,
        }
        for name, component in (
            ("asr", metadata.asr),
            ("alignment", metadata.alignment),
            ("diarization", metadata.diarization),
        )
    }


def _add_counts(left: ErrorCounts, right: ErrorCounts) -> ErrorCounts:
    return ErrorCounts(
        substitutions=left.substitutions + right.substitutions,
        deletions=left.deletions + right.deletions,
        insertions=left.insertions + right.insertions,
        reference_words=left.reference_words + right.reference_words,
    )


def _micro(results: list[dict[str, object]]) -> dict[str, object]:
    wer = ErrorCounts(0, 0, 0, 0)
    attributed = ErrorCounts(0, 0, 0, 0)
    miss = false_alarm = confusion = reference_seconds = role_correct_seconds = 0.0
    for result in results:
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        wer_metrics = metrics["wer"]
        attributed_metrics = metrics["speaker_attributed_wer"]
        der_metrics = metrics["der"]
        assert isinstance(wer_metrics, dict)
        assert isinstance(attributed_metrics, dict)
        assert isinstance(der_metrics, dict)
        wer = _add_counts(wer, _counts_from_json(wer_metrics))
        attributed = _add_counts(attributed, _counts_from_json(attributed_metrics))
        miss += float(der_metrics["miss_seconds"])
        false_alarm += float(der_metrics["false_alarm_seconds"])
        confusion += float(der_metrics["confusion_seconds"])
        current_reference_seconds = float(der_metrics["reference_speaker_seconds"])
        reference_seconds += current_reference_seconds
        role_correct_seconds += float(metrics["time_weighted_role_accuracy"]) * current_reference_seconds
    return {
        "wer": _counts_json(wer),
        "der": {
            "der": (miss + false_alarm + confusion) / reference_seconds if reference_seconds else 0.0,
            "miss_seconds": miss,
            "false_alarm_seconds": false_alarm,
            "confusion_seconds": confusion,
            "reference_speaker_seconds": reference_seconds,
        },
        "time_weighted_role_accuracy": role_correct_seconds / reference_seconds if reference_seconds else 0.0,
        "speaker_attributed_wer": _counts_json(attributed),
    }


def _required_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} должен быть целым числом")
    return value


def _counts_from_json(value: dict[str, object]) -> ErrorCounts:
    return ErrorCounts(
        substitutions=_required_int(value.get("substitutions"), field="substitutions"),
        deletions=_required_int(value.get("deletions"), field="deletions"),
        insertions=_required_int(value.get("insertions"), field="insertions"),
        reference_words=_required_int(value.get("reference_words"), field="reference_words"),
    )


def _evaluate_entry(client: httpx.Client, endpoint: str, entry: ManifestEntry) -> dict[str, object]:
    content_type = _CONTENT_TYPES[entry.raw["format"]]
    started = time.monotonic()
    with entry.path.open("rb") as audio:
        response = client.post(
            endpoint,
            files={"file": (entry.path.name, audio, content_type)},
        )
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    if response.status_code == 502:
        raise CanonicalEvaluationFailure(status_code=502, reason="provider_failure")
    if response.status_code == 409:
        raise CanonicalEvaluationFailure(status_code=409, reason="role_resolution_required")
    if response.status_code != 200:
        raise CanonicalEvaluationFailure(status_code=response.status_code, reason="canonical_service_failure")
    try:
        canonical = SpeechTranscriptionResponse.model_validate_json(response.content)
    except (ValueError, TypeError) as error:
        raise CanonicalEvaluationFailure(status_code=200, reason="invalid_canonical_response") from error
    reference = load_segments(entry.root / str(entry.raw["reference_path"]))
    hypothesis = _segments(canonical)
    wer = corpus_wer(reference, hypothesis)
    der = diarization_error_rate(reference, hypothesis)
    attributed = speaker_attributed_wer(reference, hypothesis)
    transcript = canonical.transcript
    return {
        "id": entry.identifier,
        "audio_sha256": str(entry.raw["sha256"]),
        "reference_sha256": str(entry.raw["reference_sha256"]),
        "hypothesis_sha256": hashlib.sha256(
            "\n".join(segment.text for segment in transcript.segments).encode("utf-8")
        ).hexdigest(),
        "transcript_revision": transcript.revision,
        "component_revisions": _component_revisions(canonical),
        "latency_ms": latency_ms,
        "metrics": {
            "wer": _counts_json(wer),
            "der": der,
            "time_weighted_role_accuracy": time_weighted_role_accuracy(reference, hypothesis),
            "speaker_attributed_wer": _counts_json(attributed),
        },
    }


def evaluate(arguments: argparse.Namespace) -> tuple[int, dict[str, object]]:
    entries = validate_manifest(arguments.manifest, require_release_corpus=True)
    endpoint = _endpoint(arguments.base_url)
    scored_entries = tuple(entry for entry in entries if entry.kind == "speech_reference")
    results: list[dict[str, object]] = []
    with httpx.Client(timeout=arguments.timeout_seconds, follow_redirects=False, trust_env=False) as client:
        for entry in scored_entries:
            try:
                results.append(_evaluate_entry(client, endpoint, entry))
            except CanonicalEvaluationFailure as error:
                return 1, {
                    "schema_version": 1,
                    "kind": "canonical-speech-evaluation",
                    "canonical_speech_path": True,
                    "status": "failed",
                    "manifest_sha256": _sha256(arguments.manifest),
                    "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
                    "completed_files": len(results),
                    "failure": {"status_code": error.status_code, "reason": error.reason},
                }
    return 0, {
        "schema_version": 1,
        "kind": "canonical-speech-evaluation",
        "canonical_speech_path": True,
        "status": "completed",
        "manifest_sha256": _sha256(arguments.manifest),
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "files": results,
        "micro": _micro(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("test_data/manifest.yaml"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds должен быть положительным")
    status, result = evaluate(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(arguments.output)}, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
