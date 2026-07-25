"""Fail-closed release readiness gates with typed evidence verification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mtbank_ai.release.evidence import EvidenceValidationError, validate_evidence
from mtbank_ai.release.model_manifest import ModelRegistry, SpeechModelManifest

_REQUIRED_CLOUD_VARIABLES = (
    "MTBANK_AGENT_RUNTIME__GATEWAY__BASE_URL",
    "MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY",
    "MTBANK_AGENT_RUNTIME__GATEWAY__MODELS__DEFAULT_MODEL",
)
_EVIDENCE_FILES = (
    ("real_agent_traces", "real-agent-traces.json", "real-agent-traces"),
    ("gpu_benchmark", "gpu-benchmark.json", "gpu-benchmark"),
    ("grafana_browser_proof", "grafana-browser-proof.json", "grafana-browser-proof"),
    ("websocket_gpu_p95", "websocket-gpu-p95.json", "websocket-gpu-p95"),
    ("canonical_app_image", "canonical-app-image.json", "canonical-app-image"),
)
RELEASE_GATE_IDS = (
    "licensed_corpus",
    "independent_external_attestation",
    "local_model_artifacts",
    "cloud_gateway_credentials",
    *(item[0] for item in _EVIDENCE_FILES),
)


@dataclass(frozen=True, slots=True)
class ReleaseGateContext:
    root: Path
    environment: Mapping[str, str]

    @classmethod
    def from_process(cls, root: Path) -> ReleaseGateContext:
        return cls(root=root, environment=os.environ)


@dataclass(frozen=True, slots=True)
class GateResult:
    id: str
    status: str
    reason: str
    evidence_path: str | None = None
    sha256: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {"id": self.id, "status": self.status, "reason": self.reason}
        if self.evidence_path is not None:
            result["evidence_path"] = self.evidence_path
        if self.sha256 is not None:
            result["sha256"] = self.sha256
        return result


def evaluate_release_gate(context: ReleaseGateContext) -> dict[str, object]:
    """Returns a manifest passable only with verified typed proof for this checkout."""

    current_code_sha = _current_code_sha(context)
    results = (
        _licensed_corpus(context, current_code_sha),
        _independent_external_attestation(),
        _model_artifacts(context, current_code_sha),
        _cloud_credentials(context),
        *(
            _evidence_file(context, gate_id, filename, kind, current_code_sha)
            for gate_id, filename, kind in _EVIDENCE_FILES
        ),
    )
    blocked = [result.id for result in results if result.status != "passed"]
    return {
        "schema_version": "1",
        "code_sha": current_code_sha,
        "status": "blocked" if blocked else "passed",
        "blocked": blocked,
        "gates": [result.as_dict() for result in results],
    }


def release_gate_ids() -> tuple[str, ...]:
    """Returns the versioned authoritative release-gate identifiers."""

    return RELEASE_GATE_IDS


def require_real_llm_environment(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Lists required cloud configuration without disclosing any value."""

    return tuple(name for name in _REQUIRED_CLOUD_VARIABLES if not environment.get(name, "").strip())


def _licensed_corpus(context: ReleaseGateContext, current_code_sha: str | None) -> GateResult:
    configured = context.environment.get("MTBANK_LICENSED_CORPUS_MANIFEST", "").strip()
    if not configured:
        return GateResult(
            "licensed_corpus",
            "blocked",
            "MTBANK_LICENSED_CORPUS_MANIFEST не задан; transport-only fixtures не являются corpus.",
        )
    path = Path(configured)
    return _typed_evidence_file(context, "licensed_corpus", path, "licensed-corpus-manifest", current_code_sha)


def _independent_external_attestation() -> GateResult:
    """Локальные hashes не могут подтвердить external независимую проверку."""

    return GateResult(
        "independent_external_attestation",
        "blocked",
        "Требуется отдельное external независимое подтверждение; declared digest и reviewer reference hash "
        "его не заменяют.",
    )


def _model_artifacts(context: ReleaseGateContext, current_code_sha: str | None) -> GateResult:
    manifest_path = context.root / "models" / "manifest.json"
    artifacts = context.root / "models" / "artifacts"
    evidence_path = context.root / "release-evidence" / "local-model-artifacts.json"
    try:
        manifest = SpeechModelManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        registry = ModelRegistry(artifact_root=artifacts, manifest=manifest)
        if not registry.verify_ready():
            raise ValueError("artifact tree hash mismatch")
    except (OSError, ValueError):
        return GateResult(
            "local_model_artifacts",
            "blocked",
            "runtime manifest schema v3 или immutable ASR/diarization artifacts не прошли validation.",
        )
    if current_code_sha is None:
        return GateResult("local_model_artifacts", "blocked", "не удалось определить текущий Git code SHA.")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        validate_evidence(evidence, expected_kind="local-model-artifacts", expected_code_sha=current_code_sha)
        provenance = evidence["provenance"]
        metrics = evidence["metrics"]
        assert isinstance(provenance, Mapping) and isinstance(metrics, Mapping)
        expected = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "asr_artifact_sha256": manifest.asr.artifact_sha256,
            "diarization_artifact_sha256": manifest.diarization.artifact_sha256,
            "asr_model_revision": manifest.asr.model_revision,
            "diarization_model_revision": manifest.diarization.model_revision,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise EvidenceValidationError("model provenance binding mismatch")
        asr_file_count = _artifact_file_count(registry.artifact_path(manifest.asr))
        diarization_file_count = _artifact_file_count(registry.artifact_path(manifest.diarization))
        if (
            metrics.get("asr_file_count") != asr_file_count
            or metrics.get("diarization_file_count") != diarization_file_count
            or metrics.get("artifact_count") != 2
        ):
            raise EvidenceValidationError("model artifact count binding mismatch")
    except (OSError, json.JSONDecodeError, EvidenceValidationError, AssertionError):
        return GateResult(
            "local_model_artifacts",
            "blocked",
            "local-model-artifacts.json не прошло typed runtime manifest/artifact binding validation.",
        )
    return _passed_file("local_model_artifacts", evidence_path)


def _artifact_file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def _cloud_credentials(context: ReleaseGateContext) -> GateResult:
    missing = require_real_llm_environment(context.environment)
    if missing:
        return GateResult(
            "cloud_gateway_credentials",
            "blocked",
            "не заданы обязательные cloud gateway/model credentials: " + ", ".join(missing),
        )
    return GateResult("cloud_gateway_credentials", "passed", "cloud gateway/model credentials настроены.")


def _evidence_file(
    context: ReleaseGateContext,
    gate_id: str,
    filename: str,
    kind: str,
    current_code_sha: str | None,
) -> GateResult:
    path = context.root / "release-evidence" / filename
    return _typed_evidence_file(context, gate_id, path, kind, current_code_sha)


def _typed_evidence_file(
    context: ReleaseGateContext,
    gate_id: str,
    path: Path,
    kind: str,
    current_code_sha: str | None,
) -> GateResult:
    del context
    if not path.is_file():
        return GateResult(gate_id, "blocked", f"не найдено privacy-safe evidence: {path.name}.")
    if current_code_sha is None:
        return GateResult(gate_id, "blocked", "не удалось определить текущий Git code SHA.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_evidence(payload, expected_kind=kind, expected_code_sha=current_code_sha)
    except (OSError, json.JSONDecodeError, EvidenceValidationError):
        return GateResult(
            gate_id,
            "blocked",
            f"evidence {path.name} не прошло typed provenance/hash/code-SHA validation.",
        )
    return _passed_file(gate_id, path)


def _current_code_sha(context: ReleaseGateContext) -> str | None:
    configured = context.environment.get("MTBANK_RELEASE_CODE_SHA", "").strip()
    try:
        completed = subprocess.run(
            ("git", "-C", str(context.root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    git_sha = completed.stdout.strip()
    if configured and configured != git_sha:
        return None
    return git_sha


def _passed_file(gate_id: str, path: Path) -> GateResult:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return GateResult(gate_id, "passed", "evidence предоставлено.", str(path), digest)
