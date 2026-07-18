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


def _model_artifacts(context: ReleaseGateContext, current_code_sha: str | None) -> GateResult:
    manifest = context.root / "models" / "manifest.json"
    artifacts = context.root / "models" / "artifacts"
    if not artifacts.is_dir() or not any(artifacts.iterdir()):
        return GateResult(
            "local_model_artifacts",
            "blocked",
            "локальные immutable model artifacts не предоставлены.",
        )
    return _typed_evidence_file(context, "local_model_artifacts", manifest, "local-model-artifacts", current_code_sha)


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
