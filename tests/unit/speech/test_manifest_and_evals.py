from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from mtbank_ai.speech.dataset import ManifestError, validate_manifest
from scripts.evaluate_speech import (
    Segment,
    corpus_wer,
    diarization_error_rate,
    speaker_attributed_wer,
    time_weighted_role_accuracy,
)
from services.speech.settings import SpeechAccessSettings, SpeechRuntimeSettings, SpeechSettings
from tests.unit.speech._helpers import make_registry

ROOT = Path(__file__).parents[3]


def test_manifest_contains_release_ready_speech_corpus_and_transport_fixtures() -> None:
    manifest = ROOT / "test_data" / "manifest.yaml"

    entries = validate_manifest(manifest, require_release_corpus=True)

    assert len(entries) == 8
    assert sum(entry.kind == "speech_reference" for entry in entries) == 5
    assert sum(entry.kind == "transport_only" for entry in entries) == 3
    assert sum(entry.duration_seconds for entry in entries if entry.kind == "speech_reference") >= 300
    assert any(entry.sample_rate_hz == 8000 for entry in entries if entry.kind == "speech_reference")


def test_external_wer_baseline_is_bound_to_current_synthetic_manifest() -> None:
    manifest_path = ROOT / "test_data" / "manifest.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = json.loads(
        (ROOT / "test_data" / "evaluations" / "groq-whisper-large-v3.json").read_text(encoding="utf-8")
    )
    speech_entries = {entry["id"]: entry for entry in manifest["entries"] if entry["kind"] == "speech_reference"}

    assert baseline["canonical_speech_path"] is False
    assert baseline["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert {item["id"] for item in baseline["files"]} == set(speech_entries)
    for item in baseline["files"]:
        entry = speech_entries[item["id"]]
        assert item["audio_sha256"] == entry["sha256"]
        assert item["reference_sha256"] == entry["reference_sha256"]
    micro = baseline["micro"]
    assert micro["reference_words"] > 0
    assert micro["wer"] == pytest.approx(
        (micro["substitutions"] + micro["deletions"] + micro["insertions"]) / micro["reference_words"]
    )


def _manifest_payload() -> dict[str, object]:
    return json.loads((ROOT / "test_data" / "manifest.yaml").read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_release_gate_rejects_duplicate_silence_entries(tmp_path: Path) -> None:
    payload = _manifest_payload()
    dataset = payload["dataset"]
    entries = payload["entries"]
    assert isinstance(dataset, dict)
    assert isinstance(entries, list)
    dataset["status"] = "release_ready"
    for index in range(4):
        duplicate = copy.deepcopy(entries[0])
        duplicate["id"] = f"duplicate-silence-{index}"
        entries.append(duplicate)

    with pytest.raises(ManifestError, match="audio paths должны быть уникальны"):
        validate_manifest(_write_manifest(tmp_path, payload), require_release_corpus=True)


def test_manifest_probes_actual_audio_properties(tmp_path: Path) -> None:
    payload = _manifest_payload()
    entries = payload["entries"]
    assert isinstance(entries, list)
    fixture = tmp_path / "fixture.wav"
    fixture.write_bytes((ROOT / "test_data" / "transport" / "silence-16k.wav").read_bytes())
    entries[0]["path"] = fixture.name
    entries[0]["sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    entries[0]["duration_seconds"] = 60.0

    with pytest.raises(ManifestError, match="duration_seconds не совпадает с фактическим аудио"):
        validate_manifest(_write_manifest(tmp_path, payload), require_release_corpus=False)


def test_release_gate_rejects_hashed_empty_reference_schema(tmp_path: Path) -> None:
    source = ROOT / "test_data" / "transport" / "silence-16k.wav"
    fixture = tmp_path / "fixture.wav"
    fixture.write_bytes(source.read_bytes())
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    entry = {
        "id": "manufactured-release-entry",
        "kind": "speech_reference",
        "path": fixture.name,
        "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "format": "wav",
        "sample_rate_hz": 16000,
        "channels": 1,
        "duration_seconds": 1.0,
        "license": "LicenseRef-test",
        "provenance": "Тестовый fixture для проверки release gate.",
        "eligible_for": ["wer", "der", "role_accuracy", "speaker_attributed_wer"],
        "excluded_from": [],
        "reference_path": reference.name,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "speaker_count": 2,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": {"status": "release_ready"},
        "entries": [entry],
    }

    with pytest.raises(ManifestError, match="reference требует точную schema с segments"):
        validate_manifest(_write_manifest(tmp_path, payload), require_release_corpus=True)


def test_eval_math_uses_normalized_tokens_time_weighted_roles_and_speaker_attribution() -> None:
    reference = (
        Segment("a", 0.0, 1.0, "Оператор", "Добрый, день!"),
        Segment("b", 1.0, 2.0, "Клиент", "Карта"),
    )
    hypothesis = (
        Segment("a", 0.0, 1.0, "Оператор", "добрый вечер"),
        Segment("b", 1.0, 2.0, "Оператор", "кредит"),
    )

    wer = corpus_wer(reference, hypothesis)
    role_accuracy = time_weighted_role_accuracy(reference, hypothesis)
    attributed = speaker_attributed_wer(reference, hypothesis)

    assert (wer.substitutions, wer.deletions, wer.insertions, wer.reference_words) == (2, 0, 0, 3)
    assert wer.rate == pytest.approx(2 / 3)
    assert role_accuracy == pytest.approx(0.5)
    assert attributed.rate == pytest.approx(1.0)


def test_corpus_wer_ignores_segment_identifiers_and_segmentation() -> None:
    reference = (
        Segment("reference-a", 0.0, 1.0, "Оператор", "Добрый день"),
        Segment("reference-b", 1.0, 2.0, "Клиент", "Спасибо"),
    )
    hypothesis = (
        Segment("e03f3d40-0829-4dbf-9de4-9701632a8d42", 0.0, 0.5, "Оператор", "добрый"),
        Segment("a4303945-b974-411d-bc08-bcc8d2c16e82", 0.5, 2.0, "Клиент", "день спасибо"),
    )

    wer = corpus_wer(reference, hypothesis)

    assert wer.errors == 0
    assert wer.reference_words == 3


def test_der_permits_label_permutation_but_counts_time_not_segment_count() -> None:
    reference = (
        Segment("a", 0.0, 1.0, "SPEAKER_00", "a"),
        Segment("b", 1.0, 3.0, "SPEAKER_01", "b"),
    )
    hypothesis = (
        Segment("x", 0.0, 1.0, "B", "a"),
        Segment("y", 1.0, 3.0, "A", "b"),
    )

    metrics = diarization_error_rate(reference, hypothesis)

    assert metrics["der"] == 0.0
    assert metrics["reference_speaker_seconds"] == pytest.approx(3.0)


def test_model_registry_is_fail_closed_for_tampered_local_artifact(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)
    (tmp_path / "artifacts" / "diarization" / "artifact.bin").write_bytes(b"tampered")

    assert not registry.verify_ready()


def test_model_registry_rechecks_artifacts_after_a_successful_readiness_probe(tmp_path: Path) -> None:
    registry, _ = make_registry(tmp_path)

    assert registry.verify_ready()
    (tmp_path / "artifacts" / "diarization" / "artifact.bin").write_bytes(b"tampered")
    assert not registry.verify_ready()


def test_typed_speech_settings_requires_explicit_access_mode() -> None:
    with pytest.raises(ValidationError, match="access"):
        SpeechSettings.model_validate({"runtime": SpeechRuntimeSettings().model_dump()})

    settings = SpeechSettings.model_validate(
        {"runtime": SpeechRuntimeSettings().model_dump(), "access": {"mode": "internal"}}
    )

    assert settings.groq is None
    assert settings.access == SpeechAccessSettings(mode="internal")
    assert settings.faster_whisper.model_id == "dropbox-dash/faster-whisper-large-v3-turbo"


def test_typed_speech_settings_requires_groq_only_for_enabled_streaming() -> None:
    with pytest.raises(ValidationError, match="Groq"):
        SpeechSettings.model_validate(
            {
                "runtime": SpeechRuntimeSettings().model_dump(),
                "streaming": {"enabled": True},
                "access": {"mode": "internal"},
            }
        )


def test_speech_images_profiles_and_lock_are_staticly_pinned() -> None:
    cpu = (ROOT / "docker" / "speech.cpu.Dockerfile").read_text(encoding="utf-8")
    gpu = (ROOT / "docker" / "speech.gpu.Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    gpu_compose = (ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    runpod_compose = (ROOT / "docker-compose.runpod.yml").read_text(encoding="utf-8")
    runpod_env = (ROOT / "deploy" / "runpod" / "env.example").read_text(encoding="utf-8")
    runpod_readme = (ROOT / "deploy" / "runpod" / "README.md").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "services" / "speech" / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "services" / "speech" / "uv.lock").read_text(encoding="utf-8"))

    expected_dependencies = {
        "faster-whisper==1.2.1",
        "fastapi==0.139.0",
        "httpx==0.28.1",
        "pyannote.audio==4.0.7",
        "torch==2.8.0",
        "websockets==16.0",
    }
    assert expected_dependencies.issubset(project["project"]["dependencies"])
    assert project["tool"]["uv"] == {"package": False}
    locked = {item["name"]: item["version"] for item in lock["package"]}
    assert locked["pyannote-audio"] == "4.0.7"
    assert locked["faster-whisper"] == "1.2.1"
    assert "whisperx" not in locked
    dependency_manifest_path = ROOT / "services" / "speech" / "dependency-manifest.json"
    dependency_manifest = json.loads(dependency_manifest_path.read_text(encoding="utf-8"))
    assert dependency_manifest["speech_runtime"]["faster_whisper"] == {
        "package": "faster-whisper",
        "version": "1.2.1",
        "model_id": "dropbox-dash/faster-whisper-large-v3-turbo",
        "runtime": "CTranslate2",
    }
    assert dependency_manifest["speech_runtime"]["pyannote_audio"]["model_id"] == (
        "pyannote/speaker-diarization-community-1"
    )
    for dockerfile in (cpu, gpu):
        assert "HF_HUB_OFFLINE=1" in dockerfile
        assert "TRANSFORMERS_OFFLINE=1" in dockerfile
        assert '"--workers", "1"' in dockerfile
        assert '"--ws-max-size", "65540", "--ws-max-queue", "1"' in dockerfile
        assert "HF_TOKEN" not in dockerfile
        assert "https://deb.debian.org" in dockerfile
        assert "https://security.debian.org" in dockerfile
        assert "--require-hashes" in dockerfile
    assert "  speech:\n" in compose
    assert "docker/speech.cpu.Dockerfile" in compose
    assert "MTBANK_SPEECH__ACCESS__MODE: internal" in compose
    assert 'MTBANK_SPEECH__STREAMING__ENABLED: "false"' in compose
    assert 'MTBANK_SPEECH__STREAMING__MAX_UPDATE_TEXT_BYTES: "49152"' in compose
    assert 'MTBANK_SPEECH__STREAMING__ROLLING_STEP_SECONDS: "1.5"' in compose
    assert 'MTBANK_SPEECH__STREAMING__MAX_CONCURRENT_ROLLING_CALLS: "1"' in compose
    assert "MTBANK_SPEECH__STREAMING_PATH: /v1/stream" in compose
    assert "      - application-internal" in compose
    assert '      - "8010"' in compose
    assert "profiles:\n      - gpu" in gpu_compose
    assert "docker/speech.gpu.Dockerfile" in gpu_compose
    assert "MTBANK_SPEECH__ACCESS__MODE: internal" in gpu_compose
    assert "capabilities:\n                - gpu" in gpu_compose
    assert "HF_TOKEN" not in compose + gpu_compose + runpod_compose
    assert "MTBANK_SPEECH__MODE: remote_https" in runpod_compose
    assert "MTBANK_RUNPOD_SPEECH_BASE_URL" in runpod_compose
    assert "MTBANK_RUNPOD_SPEECH_BEARER_KEY" in runpod_compose
    assert "MTBANK_SPEECH__ACCESS__MODE=bearer" in runpod_env
    assert "MTBANK_SPEECH__ACCESS__BEARER_KEY" in runpod_env
    assert "MTBANK_SPEECH__ACCESS__MODE=bearer" in runpod_readme
    assert "/.pi-subagents/" in dockerignore.splitlines()
    assert "/deploy/runpod/env.local" in dockerignore.splitlines()
    assert "Docker Compose or nested Docker" in runpod_readme
    assert "before container start or CUDA warmup" in runpod_readme
    assert "Docker-ignored" in runpod_readme
    assert "image@sha256" in runpod_readme


def test_manifest_is_json_compatible_yaml_for_dependency_free_validation() -> None:
    payload = json.loads((ROOT / "test_data" / "manifest.yaml").read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1


def test_websocket_overlay_requires_explicit_origin_and_enables_both_boundaries() -> None:
    overlay = (ROOT / "docker-compose.websocket.yml").read_text(encoding="utf-8")

    assert 'MTBANK_WEBSOCKET__ENABLED: "true"' in overlay
    assert 'MTBANK_SPEECH__STREAMING__ENABLED: "true"' in overlay
    assert "${MTBANK_WEBSOCKET_ALLOWED_ORIGIN:?set MTBANK_WEBSOCKET_ALLOWED_ORIGIN in .env}" in overlay
    assert "ports:" not in overlay
