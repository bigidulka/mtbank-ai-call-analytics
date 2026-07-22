"""Typed, privacy-safe release evidence serialization and verification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

_HASH_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_KEY_NORMALIZER: Final = re.compile(r"[^a-z0-9]+")
_FORBIDDEN_KEY_PARTS: Final = (
    "apikey",
    "audio",
    "authorization",
    "cookie",
    "credential",
    "header",
    "password",
    "prompt",
    "providerresponse",
    "response",
    "secret",
    "session",
    "setcookie",
    "token",
    "transcript",
)
_PROVIDER_REQUEST_ID_KEYS: Final = frozenset({"providerrequestid", "providerrequestids"})
_SAFE_REVISION_KEYS: Final = frozenset({"promptbundlehash"})

_REAL_TRACE_PROVENANCE: Final = frozenset(
    {
        "invocation_nonce_sha256",
        "model_id",
        "model_revision",
        "provider_id",
        "run_endpoint_sha256",
        "run_id_sha256",
        "trace_artifact_sha256",
    }
)
_EVIDENCE_SCHEMAS: Final = {
    "licensed-corpus-manifest": (
        frozenset({"license_document_sha256", "manifest_source_sha256", "reviewer_id_sha256", "dataset_revision"}),
        frozenset({"reference_transcript_count", "role_label_count"}),
        frozenset({"dataset_id"}),
    ),
    "local-model-artifacts": (
        frozenset(
            {
                "manifest_sha256",
                "asr_artifact_sha256",
                "diarization_artifact_sha256",
                "asr_model_revision",
                "diarization_model_revision",
                "reviewer_id_sha256",
            }
        ),
        frozenset({"artifact_count", "asr_file_count", "diarization_file_count"}),
        frozenset({"model_set_id"}),
    ),
    "real-agent-traces": (
        _REAL_TRACE_PROVENANCE,
        frozenset({"retrieval_calls", "terminal_submissions"}),
        frozenset({"agent_ids", "provider_request_ids_sha256"}),
    ),
    "gpu-benchmark": (
        frozenset(
            {
                "image_digest",
                "model_manifest_sha256",
                "runner_id_sha256",
                "runtime_attestation_sha256",
                "workload_revision",
            }
        ),
        frozenset({"p50_ms", "p95_ms", "throughput_per_second"}),
        frozenset(),
    ),
    "grafana-browser-proof": (
        frozenset({"browser_harness_revision", "dashboard_revision", "endpoint_sha256", "screenshot_sha256"}),
        frozenset({"panel_count"}),
        frozenset(),
    ),
    "websocket-gpu-p95": (
        frozenset(
            {
                "image_digest",
                "model_manifest_sha256",
                "runner_id_sha256",
                "runtime_attestation_sha256",
                "workload_revision",
            }
        ),
        frozenset({"p95_ms", "session_count"}),
        frozenset(),
    ),
    "canonical-app-image": (
        frozenset({"dockerfile_sha256", "image_digest", "lock_sha256", "runner_id_sha256"}),
        frozenset({"build_duration_seconds"}),
        frozenset(),
    ),
}


class EvidenceValidationError(ValueError):
    """Evidence is incomplete, tampered, or unsuitable for a release gate."""


def canonical_json_bytes(value: object) -> bytes:
    """Returns the stable serialization used for evidence integrity hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_evidence(value: object) -> object:
    """Drops content-bearing and credential-bearing fields at every nesting depth."""

    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str):
                raise EvidenceValidationError("ключ evidence должен быть строкой")
            normalized = _normalize_key(raw_key)
            if normalized in _PROVIDER_REQUEST_ID_KEYS:
                result[f"{raw_key}_sha256"] = _hash_provider_request_ids(raw_value)
                continue
            if normalized not in _SAFE_REVISION_KEYS and any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                continue
            result[raw_key] = sanitize_evidence(raw_value)
        return result
    if isinstance(value, tuple | list):
        return [sanitize_evidence(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise EvidenceValidationError("evidence содержит неподдерживаемый тип")


def export_evidence(
    *,
    kind: str,
    code_sha: str,
    provenance: Mapping[str, object],
    metrics: Mapping[str, object],
    observations: Mapping[str, object],
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Builds one schema-checked, content-free evidence record for a release artifact."""

    created_at = generated_at or datetime.now(UTC)
    if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(created_at):
        raise EvidenceValidationError("generated_at должен быть UTC")
    payload = {
        "schema_version": "1",
        "kind": kind,
        "code_sha": code_sha,
        "generated_at": created_at.isoformat().replace("+00:00", "Z"),
        "provenance": dict(provenance),
        "metrics": dict(metrics),
        "observations": dict(observations),
    }
    _validate_shape(payload, expected_kind=kind, expected_code_sha=code_sha)
    return {**payload, "sha256": sha256(payload)}


def validate_evidence(payload: object, *, expected_kind: str, expected_code_sha: str) -> None:
    """Fail closed unless a typed artifact has a canonical integrity hash and current code SHA."""

    if not isinstance(payload, Mapping):
        raise EvidenceValidationError("evidence должен быть JSON object")
    expected_keys = {
        "schema_version",
        "kind",
        "code_sha",
        "generated_at",
        "provenance",
        "metrics",
        "observations",
        "sha256",
    }
    if set(payload) != expected_keys:
        raise EvidenceValidationError("evidence содержит недопустимые поля")
    provided_hash = payload["sha256"]
    if not isinstance(provided_hash, str) or not _HASH_RE.fullmatch(provided_hash):
        raise EvidenceValidationError("evidence sha256 имеет недопустимый формат")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    if sha256(unsigned) != provided_hash:
        raise EvidenceValidationError("evidence sha256 не совпадает с canonical payload")
    _validate_shape(unsigned, expected_kind=expected_kind, expected_code_sha=expected_code_sha)


def _validate_shape(payload: Mapping[str, object], *, expected_kind: str, expected_code_sha: str) -> None:
    if payload.get("schema_version") != "1":
        raise EvidenceValidationError("неподдерживаемая evidence schema")
    if payload.get("kind") != expected_kind or expected_kind not in _EVIDENCE_SCHEMAS:
        raise EvidenceValidationError("evidence kind не соответствует release gate")
    if payload.get("code_sha") != expected_code_sha or not _nonempty_string(expected_code_sha):
        raise EvidenceValidationError("evidence code_sha не соответствует текущему коду")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise EvidenceValidationError("evidence generated_at должен быть UTC")
    try:
        datetime.fromisoformat(generated_at.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise EvidenceValidationError("evidence generated_at имеет недопустимый формат") from error
    provenance_keys, metric_keys, observation_keys = _EVIDENCE_SCHEMAS[expected_kind]
    _validate_section(payload.get("provenance"), provenance_keys, "provenance")
    _validate_section(payload.get("metrics"), metric_keys, "metrics")
    _validate_section(payload.get("observations"), observation_keys, "observations")
    _validate_safe_values(payload.get("provenance"))
    _validate_safe_values(payload.get("metrics"))
    _validate_safe_values(payload.get("observations"))
    _validate_kind_values(expected_kind, payload)


def _validate_section(value: object, expected_keys: frozenset[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise EvidenceValidationError(f"evidence {name} не соответствует typed schema")


def _validate_safe_values(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceValidationError("ключ evidence должен быть строкой")
            normalized = _normalize_key(key)
            if normalized not in _SAFE_REVISION_KEYS and any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise EvidenceValidationError("evidence содержит content или credential field")
            _validate_safe_values(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _validate_safe_values(item)
        return
    if value is None or isinstance(value, str | int | float | bool):
        return
    raise EvidenceValidationError("evidence содержит неподдерживаемый тип")


def _validate_kind_values(kind: str, payload: Mapping[str, object]) -> None:
    provenance = payload["provenance"]
    metrics = payload["metrics"]
    observations = payload["observations"]
    assert isinstance(provenance, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(observations, Mapping)
    for key, value in provenance.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or not _HASH_RE.fullmatch(value)):
            raise EvidenceValidationError(f"evidence {key} должен быть SHA-256")
        if not key.endswith("_sha256") and not _nonempty_string(value):
            raise EvidenceValidationError(f"evidence {key} обязателен")
    for key, value in metrics.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise EvidenceValidationError(f"evidence metric {key} должен быть неотрицательным числом")
    if kind == "real-agent-traces":
        agent_ids = observations["agent_ids"]
        request_ids = observations["provider_request_ids_sha256"]
        if agent_ids != ["classifier", "compliance", "quality", "summarizer"]:
            raise EvidenceValidationError("evidence trace не подтверждает четыре distinct agents")
        if not isinstance(request_ids, list) or len(request_ids) != 4 or len(set(request_ids)) != 4:
            raise EvidenceValidationError("evidence trace не подтверждает distinct provider request IDs")
        if not all(isinstance(value, str) and _HASH_RE.fullmatch(value) for value in request_ids):
            raise EvidenceValidationError("evidence provider request IDs должны быть SHA-256")
        if metrics["retrieval_calls"] < 1 or metrics["terminal_submissions"] != 4:
            raise EvidenceValidationError("evidence trace не подтверждает retrieval и terminal submissions")


def _normalize_key(value: str) -> str:
    return _KEY_NORMALIZER.sub("", value.casefold())


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _hash_provider_request_ids(value: object) -> str | list[str]:
    if isinstance(value, str):
        return sha256_text(value)
    if isinstance(value, tuple | list) and all(isinstance(item, str) for item in value):
        return [sha256_text(item) for item in value]
    raise EvidenceValidationError("provider request ID должен быть строкой или массивом строк")
