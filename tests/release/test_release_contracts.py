from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mtbank_ai.release.gates as release_gates
from mtbank_ai.release.evidence import (
    EvidenceValidationError,
    export_evidence,
    sanitize_evidence,
    sha256,
    validate_evidence,
)
from mtbank_ai.release.gates import ReleaseGateContext, evaluate_release_gate, require_real_llm_environment
from services.speech.manifest import ModelArtifact, ModelRegistry, SpeechModelManifest, artifact_tree_sha256

ROOT = Path(__file__).parents[2]
CODE_SHA = "a" * 40


def _load_real_e2e() -> object:
    path = ROOT / "scripts" / "release_real_e2e.py"
    spec = importlib.util.spec_from_file_location("release_real_e2e", path)
    if spec is None or spec.loader is None:
        raise AssertionError("release real E2E harness unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_evidence(*, code_sha: str = CODE_SHA) -> dict[str, object]:
    return export_evidence(
        kind="real-agent-traces",
        code_sha=code_sha,
        provenance={
            "invocation_nonce_sha256": "1" * 64,
            "model_id": "model-v1",
            "model_revision": "model/revision-1",
            "provider_id": "openai-compatible",
            "run_endpoint_sha256": "2" * 64,
            "run_id_sha256": "3" * 64,
            "trace_artifact_sha256": "4" * 64,
        },
        metrics={"retrieval_calls": 1, "terminal_submissions": 4},
        observations={
            "agent_ids": ["classifier", "compliance", "quality", "summarizer"],
            "provider_request_ids_sha256": ["5" * 64, "6" * 64, "7" * 64, "8" * 64],
        },
        generated_at=datetime(2026, 7, 17, tzinfo=UTC),
    )


def _local_model_evidence_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    artifacts = tmp_path / "models" / "artifacts"
    asr = artifacts / "asr"
    diarization = artifacts / "diarization"
    asr.mkdir(parents=True)
    diarization.mkdir(parents=True)
    (asr / "model.bin").write_bytes(b"asr")
    (diarization / "model.bin").write_bytes(b"diarization")
    manifest = SpeechModelManifest(
        asr=ModelArtifact(
            package="faster-whisper",
            package_version="1.2.1",
            model_id="asr",
            model_revision="asr-r1",
            relative_path="asr",
            artifact_sha256=artifact_tree_sha256(asr),
        ),
        diarization=ModelArtifact(
            package="pyannote.audio",
            package_version="4.0.7",
            model_id="diarization",
            model_revision="diarization-r1",
            relative_path="diarization",
            artifact_sha256=artifact_tree_sha256(diarization),
        ),
    )
    manifest_path = tmp_path / "models" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    registry = ModelRegistry(artifact_root=artifacts, manifest=manifest)
    evidence = export_evidence(
        kind="local-model-artifacts",
        code_sha=CODE_SHA,
        provenance={
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "asr_artifact_sha256": manifest.asr.artifact_sha256,
            "diarization_artifact_sha256": manifest.diarization.artifact_sha256,
            "asr_model_revision": manifest.asr.model_revision,
            "diarization_model_revision": manifest.diarization.model_revision,
            "reviewer_reference_sha256": "1" * 64,
        },
        metrics={"artifact_count": 2, "asr_file_count": 1, "diarization_file_count": 1},
        observations={"model_set_id": "local-test"},
    )
    del registry
    return tmp_path, evidence


def _model_gate(context_root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(release_gates, "_current_code_sha", lambda _: CODE_SHA)
    return evaluate_release_gate(ReleaseGateContext(root=context_root, environment={}))


def _local_model_gate_result(manifest: dict[str, object]) -> dict[str, object]:
    gates = manifest["gates"]
    assert isinstance(gates, list)
    gate = next(item for item in gates if isinstance(item, dict) and item.get("id") == "local_model_artifacts")
    assert isinstance(gate, dict)
    return gate


def test_local_model_evidence_gate_accepts_verified_manifest_artifacts_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence = _local_model_evidence_root(tmp_path)
    path = root / "release-evidence" / "local-model-artifacts.json"
    path.parent.mkdir()
    path.write_text(json.dumps(evidence), encoding="utf-8")

    manifest = _model_gate(root, monkeypatch)

    gate = _local_model_gate_result(manifest)
    assert gate["status"] == "passed"
    assert manifest["status"] == "blocked"
    gates = manifest["gates"]
    assert isinstance(gates, list)
    external = next(
        item for item in gates if isinstance(item, dict) and item.get("id") == "independent_external_attestation"
    )
    assert isinstance(external, dict)
    assert external["status"] == "blocked"


def test_local_model_evidence_gate_rejects_tampered_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, evidence = _local_model_evidence_root(tmp_path)
    evidence["provenance"]["asr_model_revision"] = "tampered"  # type: ignore[index]
    unsigned = {key: value for key, value in evidence.items() if key != "sha256"}
    evidence["sha256"] = sha256(unsigned)
    path = root / "release-evidence" / "local-model-artifacts.json"
    path.parent.mkdir()
    path.write_text(json.dumps(evidence), encoding="utf-8")

    manifest = _model_gate(root, monkeypatch)

    gate = _local_model_gate_result(manifest)
    assert gate["status"] == "blocked"


def test_local_model_evidence_gate_requires_separate_evidence_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _local_model_evidence_root(tmp_path)

    manifest = _model_gate(root, monkeypatch)

    gate = _local_model_gate_result(manifest)
    assert gate["status"] == "blocked"
    reason = gate["reason"]
    assert isinstance(reason, str)
    assert "local-model-artifacts.json" in reason


def test_redaction_drops_nested_content_credentials_and_headers() -> None:
    sanitized = sanitize_evidence(
        {
            "prompt_bundle_hash": "a" * 64,
            "app_runtime_binding_sha256": "b" * 64,
            "nested": {
                "Api-Key": "must-not-escape",
                "x_password": "must-not-escape",
                "clientCredential": "must-not-escape",
                "Authorization_Header": "must-not-escape",
                "Cookie": "must-not-escape",
                "Set-Cookie": "must-not-escape",
                "session.id": "must-not-escape",
                "raw_audio": "must-not-escape",
                "raw_transcript": "must-not-escape",
                "provider_response": "must-not-escape",
                "provider_request_ids": ["provider-request-private"],
                "accepted": True,
            },
        }
    )
    serialized = json.dumps(sanitized)

    assert "must-not-escape" not in serialized
    assert "provider-request-private" not in serialized
    assert sanitized == {
        "prompt_bundle_hash": "a" * 64,
        "app_runtime_binding_sha256": "b" * 64,
        "nested": {
            "provider_request_ids_sha256": ["5ca6ff8dd27a9ef0dfb6d76f0e410d74877d0b7eff5de22c411c94c4fc22f518"],
            "accepted": True,
        },
    }


def test_evidence_requires_kind_canonical_hash_current_code_and_typed_provenance() -> None:
    evidence = _trace_evidence()
    validate_evidence(evidence, expected_kind="real-agent-traces", expected_code_sha=CODE_SHA)

    tampered = {**evidence, "metrics": {"retrieval_calls": 2, "terminal_submissions": 4}}
    with pytest.raises(EvidenceValidationError, match="sha256"):
        validate_evidence(tampered, expected_kind="real-agent-traces", expected_code_sha=CODE_SHA)
    with pytest.raises(EvidenceValidationError, match="code_sha"):
        validate_evidence(evidence, expected_kind="real-agent-traces", expected_code_sha="b" * 40)

    fake = {key: value for key, value in evidence.items() if key != "sha256"}
    fake["kind"] = "gpu-benchmark"
    fake["sha256"] = sha256(fake)
    with pytest.raises(EvidenceValidationError, match="kind"):
        validate_evidence(fake, expected_kind="real-agent-traces", expected_code_sha=CODE_SHA)


def test_release_gate_is_blocked_without_real_infrastructure(tmp_path: Path) -> None:
    manifest = evaluate_release_gate(ReleaseGateContext(root=tmp_path, environment={}))

    assert manifest["status"] == "blocked"
    blocked = manifest["blocked"]
    assert isinstance(blocked, list)
    assert set(blocked) == {
        "licensed_corpus",
        "independent_external_attestation",
        "local_model_artifacts",
        "cloud_gateway_credentials",
        "real_agent_traces",
        "gpu_benchmark",
        "grafana_browser_proof",
        "websocket_gpu_p95",
        "canonical_app_image",
    }


def test_real_e2e_attestation_requires_nonce_endpoint_and_trace_artifact_binding() -> None:
    module = _load_real_e2e()
    trajectories = [
        {
            "agent_id": agent_id,
            "retrieval_calls": 1 if agent_id == "classifier" else 0,
            "terminal_submissions": 1,
            "provider_request_id": f"request-{agent_id}",
        }
        for agent_id in ("classifier", "quality", "compliance", "summarizer")
    ]
    attestation = {
        "schema_version": "1",
        "kind": "real-agent-run-attestation",
        "code_sha": CODE_SHA,
        "invocation_nonce": "nonce-1",
        "run_endpoint": "https://e2e.example.test/analyze",
        "run_id": "run-1",
        "provider": {
            "provider_id": "openai-compatible",
            "model_id": "model-v1",
            "model_revision": "model/revision-1",
        },
        "trace": {"artifact_sha256": sha256(trajectories), "trajectories": trajectories},
    }

    trace, provenance = module.validate_live_attestation(  # type: ignore[attr-defined]
        attestation,
        invocation_nonce="nonce-1",
        run_endpoint="https://e2e.example.test/analyze",
        code_sha=CODE_SHA,
    )
    assert trace["terminal_submissions"] == 4
    assert provenance["trace_artifact_sha256"] == sha256(trajectories)
    attestation["code_sha"] = "b" * 40
    with pytest.raises(ValueError, match="code SHA"):
        module.validate_live_attestation(  # type: ignore[attr-defined]
            attestation,
            invocation_nonce="nonce-1",
            run_endpoint="https://e2e.example.test/analyze",
            code_sha=CODE_SHA,
        )


def test_real_llm_marker_fails_in_release_mode_when_credentials_are_absent() -> None:
    environment = os.environ.copy()
    environment["MTBANK_RELEASE_GATE"] = "1"
    for name in (
        "MTBANK_AGENT_RUNTIME__GATEWAY__BASE_URL",
        "MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY",
        "MTBANK_AGENT_RUNTIME__GATEWAY__MODELS__DEFAULT_MODEL",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        (sys.executable, "-m", "pytest", "-q", "-m", "real_llm", str(Path(__file__))),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "release gate: отсутствуют" in completed.stdout


@pytest.mark.real_llm
def test_real_llm_marker_never_silently_skips_release_gate() -> None:
    missing = require_real_llm_environment(os.environ)
    if missing and os.environ.get("MTBANK_RELEASE_GATE") == "1":
        pytest.fail("release gate: отсутствуют " + ", ".join(missing))
    if missing:
        pytest.skip("normal offline suite: real cloud gateway is intentionally not configured")


@pytest.mark.gpu
def test_gpu_marker_never_silently_skips_release_gate() -> None:
    evidence = os.environ.get("MTBANK_GPU_BENCHMARK_EVIDENCE", "")
    if not evidence and os.environ.get("MTBANK_RELEASE_GATE") == "1":
        pytest.fail("release gate: отсутствует MTBANK_GPU_BENCHMARK_EVIDENCE")
    if not evidence:
        pytest.skip("normal offline suite: GPU benchmark evidence is intentionally not configured")
    assert Path(evidence).is_file()


@pytest.mark.parametrize("script", ("check_release_gate.py", "run_public_analyze_sla_benchmark.py"))
def test_installed_release_cli_help_needs_no_project_pythonpath(tmp_path: Path, script: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        (sys.executable, str(ROOT / "scripts" / script), "--help"),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_versioned_gate_ids_include_external_hard_block_without_false_evidence() -> None:
    assert set(release_gates.release_gate_ids()) == {
        "licensed_corpus",
        "independent_external_attestation",
        "local_model_artifacts",
        "cloud_gateway_credentials",
        "real_agent_traces",
        "gpu_benchmark",
        "grafana_browser_proof",
        "websocket_gpu_p95",
        "canonical_app_image",
    }
    assert not (ROOT / "release-evidence").exists()
