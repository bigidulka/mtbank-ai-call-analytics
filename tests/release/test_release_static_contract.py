from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_ci_has_offline_real_and_gpu_release_jobs() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    real = (ROOT / ".github" / "workflows" / "e2e-real.yml").read_text(encoding="utf-8")
    gpu = (ROOT / ".github" / "workflows" / "gpu-benchmark.yml").read_text(encoding="utf-8")

    for command in (
        "uv lock --check",
        "services/speech/uv.lock",
        "ruff check .",
        "pyright",
        'pytest -m "not integration and not real_llm and not gpu"',
        "docker build -f docker/app.Dockerfile",
        "check_release_static.py secrets",
    ):
        assert command in ci
    assert 'MTBANK_RELEASE_GATE: "1"' in real
    assert "pytest -m real_llm" in real
    assert "release_real_e2e.py" in real
    assert "--attestation-url" in real
    assert "MTBANK_REAL_E2E_ATTESTATION_URL" in real
    assert "MTBANK_REAL_E2E_TRACE_FILE" not in real
    assert "secrets.MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY" in real
    assert "workflow_dispatch:" in gpu
    assert "self-hosted" in gpu
    assert "run_gpu_speech_benchmark.py" in gpu
    assert "docker image inspect" in gpu
    assert "Validate generated GPU evidence for this checkout" in gpu
    assert "websocket-gpu-p95.json" in gpu
    assert "pytest -m gpu" in gpu


def test_release_docs_link_real_contracts_and_known_blockers() -> None:
    required = (
        "architecture.md",
        "api.md",
        "evals.md",
        "privacy.md",
        "operations.md",
        "threat-model.md",
        "release-checklist.md",
    )
    for filename in required:
        content = (ROOT / "docs" / filename).read_text(encoding="utf-8")
        assert "release" in content.casefold() or "Release" in content
    checklist = (ROOT / "docs" / "release-checklist.md").read_text(encoding="utf-8")
    for blocker in (
        "Licensed corpus approval",
        "Community-1 artifact",
        "Groq credentials",
        "remote raw-audio disclosure",
        "Final gateway HTTPS",
        "GPU",
        "Grafana",
        "canonical image digest",
        "Final competitor score",
    ):
        assert blocker in checklist
    assert "--allow-blocked" in checklist
    assert "diagnostic blocked-gate report" in checklist
    assert "external CI/runtime artifact" in checklist


def test_readme_discloses_asr_deviation_and_uncollected_runtime_evidence() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for value in (
        "Groq whisper-large-v3-turbo",
        "local `faster-whisper`",
        "не реализованы",
        "Нет corpus-wide canonical WER/DER/role/speaker-attributed metrics",
        "diagnostic-only",
        "не является GPU evidence",
        "run_gpu_speech_benchmark.py",
    ):
        assert value in readme
    assert "dropbox-dash/faster-whisper-large-v3-turbo" not in readme
    assert "faster-whisper medium →" not in readme
    assert "multilingual medium" not in readme
    assert "Real external Groq ASR baseline" not in readme
    assert "Real 30-секундный canonical REST E2E" not in readme


def test_dockerfiles_and_operations_document_hashed_wheelhouse_contract() -> None:
    dockerfiles = tuple(
        (ROOT / "docker" / filename).read_text(encoding="utf-8")
        for filename in ("app.Dockerfile", "speech.cpu.Dockerfile", "speech.gpu.Dockerfile")
    )
    operations = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")

    assert (ROOT / "docker" / "wheelhouse" / ".gitkeep").is_file()
    for dockerfile in dockerfiles:
        assert "COPY docker/wheelhouse /opt/wheelhouse" in dockerfile
        assert "ARG USE_WHEELHOUSE=0" in dockerfile
        assert "--no-index --find-links /opt/wheelhouse --require-hashes" in dockerfile
        assert "rm -rf /opt/wheelhouse" in dockerfile
    assert "cp /secure/reviewed-wheelhouse/*.whl docker/wheelhouse/" in operations
    assert "--build-arg USE_WHEELHOUSE=1" in operations
    assert "check_release_gate.py --allow-blocked" in operations
    assert "diagnostic only" in operations
    assert "external CI/runtime artifact" in operations


def test_compose_release_contract_keeps_cpu_gpu_monitoring_private() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    gpu = (ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")

    for service in ("api:", "speech:", "prometheus:", "tempo:", "otel-collector:", "grafana:"):
        assert service in compose
    assert "dockerfile: docker/speech.cpu.Dockerfile" in compose
    assert "dockerfile: docker/speech.gpu.Dockerfile" in gpu
    assert "capabilities:\n                - gpu" in gpu
    assert "ports:" not in gpu
