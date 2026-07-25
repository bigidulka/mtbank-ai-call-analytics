from __future__ import annotations

import json
from pathlib import Path

from pydantic import HttpUrl, SecretStr

from services.speech.manifest import ModelArtifact, ModelRegistry, SpeechModelManifest, artifact_tree_sha256
from services.speech.settings import (
    GroqTranscriptionSettings,
    RoleAgentSettings,
    SpeechAccessSettings,
    SpeechModelSettings,
    SpeechRuntimeSettings,
    SpeechSettings,
)


def make_registry(
    tmp_path: Path,
    *,
    runtime: SpeechRuntimeSettings | None = None,
) -> tuple[ModelRegistry, SpeechSettings]:
    artifact_root = tmp_path / "artifacts"
    asr = artifact_root / "asr"
    diarization = artifact_root / "diarization"
    asr.mkdir(parents=True)
    diarization.mkdir(parents=True)
    (asr / "model.bin").write_bytes(b"faster-whisper")
    (diarization / "artifact.bin").write_bytes(b"diarization")
    manifest = SpeechModelManifest(
        asr=ModelArtifact(
            package="faster-whisper",
            package_version="1.2.1",
            model_id="dropbox-dash/faster-whisper-large-v3-turbo",
            model_revision="test-asr",
            relative_path="asr",
            artifact_sha256=artifact_tree_sha256(asr),
        ),
        diarization=ModelArtifact(
            package="pyannote.audio",
            package_version="4.0.7",
            model_id="speaker-diarization-community-1",
            model_revision="test-diarization",
            relative_path="diarization",
            artifact_sha256=artifact_tree_sha256(diarization),
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    resolved_runtime = runtime or SpeechRuntimeSettings(temp_root=str(tmp_path / "work"))
    settings = SpeechSettings(
        runtime=resolved_runtime,
        groq=GroqTranscriptionSettings(api_key=SecretStr("test-groq-key")),
        role_agent=RoleAgentSettings(
            base_url=HttpUrl("https://gateway.example.test/v1"),
            api_key=SecretStr("N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"),
            model="role-model",
        ),
        access=SpeechAccessSettings(mode="internal"),
        models=SpeechModelSettings(manifest_path=str(manifest_path), artifact_root=str(artifact_root)),
    )
    return ModelRegistry.load(settings), settings
