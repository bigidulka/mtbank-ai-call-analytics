from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import provision_speech_models as provisioning
from services.speech.manifest import SpeechModelManifest

ROOT = Path(__file__).parents[3]


def test_model_sources_require_pinned_local_asr_and_community_one() -> None:
    sources = provisioning.load_model_sources(ROOT / "services" / "speech" / "model-sources.json")

    assert tuple(sources.sources) == ("asr", "diarization")
    asr = sources.sources["asr"]
    assert asr.model_id == "dropbox-dash/faster-whisper-large-v3-turbo"
    assert asr.package.name == "faster-whisper"
    assert asr.gated is False
    source = sources.sources["diarization"]
    assert source.repo_id == "pyannote/speaker-diarization-community-1"
    assert source.model_id == "pyannote/speaker-diarization-community-1"
    assert source.gated is True
    assert source.expected_artifact_content_sha256 == (
        "03130042c147ab8887d8e6f63ec6dbcc2fd970adfb4930c1cd1bf4d34647ccd4"
    )


def test_provisioning_fails_closed_before_hub_access_without_reviewed_diarization_digest(tmp_path: Path) -> None:
    source_payload = json.loads(
        (ROOT / "services" / "speech" / "model-sources.json").read_text(encoding="utf-8")
    )
    source_payload["sources"]["diarization"]["expected_artifact_content_sha256"] = None
    source_path = tmp_path / "model-sources.json"
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    calls: list[object] = []

    class ForbiddenApi:
        def model_info(self, *, repo_id: str, revision: str, token: str | None) -> object:
            del repo_id, revision, token
            raise AssertionError("Hub must not be contacted before reviewed digest validation")

    def forbidden_api_factory() -> ForbiddenApi:
        calls.append("api")
        return ForbiddenApi()

    hub = provisioning.HubFunctions(api_factory=forbidden_api_factory, snapshot_download=lambda **_: "")

    with pytest.raises(provisioning.ProvisioningError, match="reviewed pinned"):
        provisioning.provision(
            sources_path=source_path,
            artifact_root=tmp_path / "artifacts",
            output_manifest=tmp_path / "manifest.json",
            cache_dir=tmp_path / "cache",
            environment={"HF_TOKEN": "test-token"},
            hub=hub,
        )

    assert calls == []
    assert not (tmp_path / "manifest.json").exists()


def test_complete_local_manifest_rejects_legacy_alignment_field() -> None:
    artifact = {
        "package": "faster-whisper",
        "package_version": "1.2.1",
        "model_id": "dropbox-dash/faster-whisper-large-v3-turbo",
        "model_revision": "a" * 40,
        "relative_path": "asr",
        "artifact_sha256": "b" * 64,
    }
    payload = {
        "schema_version": "3",
        "asr": artifact,
        "diarization": {
            "package": "pyannote.audio",
            "package_version": "4.0.7",
            "model_id": "pyannote/speaker-diarization-community-1",
            "model_revision": "a" * 40,
            "relative_path": "diarization",
            "artifact_sha256": "b" * 64,
        },
        "alignment": {"unexpected": True},
    }

    with pytest.raises(ValueError, match="Extra inputs"):
        SpeechModelManifest.model_validate(payload)


def test_provision_cli_accepts_asr_and_diarization_components() -> None:
    parsed = provisioning._parse_arguments(
        (
            "--artifact-root",
            "artifacts",
            "--output-manifest",
            "manifest.json",
            "--cache-dir",
            "cache",
            "--component",
            "diarization",
        )
    )

    assert parsed.component == ["diarization"]
    asr = provisioning._parse_arguments(
        (
            "--artifact-root",
            "artifacts",
            "--output-manifest",
            "manifest.json",
            "--cache-dir",
            "cache",
            "--component",
            "asr",
        )
    )
    assert asr.component == ["asr"]
