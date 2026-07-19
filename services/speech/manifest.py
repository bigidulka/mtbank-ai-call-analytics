"""Verified local faster-whisper and pyannote artifacts; runtime never downloads models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from mtbank_ai.domain.base import NonEmptyId, Sha256, StrictFrozenModel
from mtbank_ai.domain.provenance import ComponentRevision
from services.speech.errors import SpeechConfigurationError
from services.speech.settings import SpeechSettings


class ModelArtifact(StrictFrozenModel):
    package: NonEmptyId
    package_version: NonEmptyId
    model_id: NonEmptyId
    model_revision: NonEmptyId
    relative_path: NonEmptyId
    artifact_sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path должен оставаться внутри artifact_root")
        return value

    def as_component_revision(self) -> ComponentRevision:
        return ComponentRevision(
            package=self.package,
            package_version=self.package_version,
            model_id=self.model_id,
            model_revision=self.model_revision,
            artifact_sha256=self.artifact_sha256,
        )


class SpeechModelManifest(StrictFrozenModel):
    schema_version: Literal["3"] = "3"
    asr: ModelArtifact
    diarization: ModelArtifact


class ModelRegistry:
    """Validates both local canonical-ASR and diarization artifacts before load."""

    def __init__(self, *, artifact_root: Path, manifest: SpeechModelManifest) -> None:
        self._artifact_root = artifact_root.resolve()
        self._manifest = manifest

    @classmethod
    def load(cls, settings: SpeechSettings) -> ModelRegistry:
        try:
            raw_manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
            manifest = SpeechModelManifest.model_validate(raw_manifest)
        except (OSError, ValueError) as error:
            raise SpeechConfigurationError("local ASR/diarization manifest is unavailable") from error
        return cls(artifact_root=settings.artifact_root, manifest=manifest)

    @property
    def manifest(self) -> SpeechModelManifest:
        return self._manifest

    def artifact_path(self, artifact: ModelArtifact) -> Path:
        target = (self._artifact_root / artifact.relative_path).resolve()
        if self._artifact_root not in target.parents:
            raise SpeechConfigurationError("model artifact path escapes artifact root")
        return target

    def asr_revision(self) -> ComponentRevision:
        return self._manifest.asr.as_component_revision()

    def diarization_revision(self) -> ComponentRevision:
        return self._manifest.diarization.as_component_revision()

    def verify_ready(self) -> bool:
        try:
            return all(
                self.artifact_path(artifact).is_dir()
                and _tree_sha256(self.artifact_path(artifact)) == artifact.artifact_sha256
                for artifact in (self._manifest.asr, self._manifest.diarization)
            )
        except (OSError, SpeechConfigurationError):
            return False


def artifact_tree_sha256(path: Path) -> str:
    """Stable tree hash used by provisioning and runtime readiness."""

    return _tree_sha256(path)


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise SpeechConfigurationError("model artifact directory is unavailable")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SpeechConfigurationError("model artifact directory is empty")
    for item in files:
        if item.is_symlink():
            raise SpeechConfigurationError("model artifacts must not contain symlinks")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
