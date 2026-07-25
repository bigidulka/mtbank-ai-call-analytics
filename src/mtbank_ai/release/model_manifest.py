"""Минимальная schema-v3 проверка immutable speech artifacts для release gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from mtbank_ai.domain.base import NonEmptyId, Sha256, StrictFrozenModel


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


class SpeechModelManifest(StrictFrozenModel):
    schema_version: Literal["3"] = "3"
    asr: ModelArtifact
    diarization: ModelArtifact


class ModelRegistry:
    """Проверяет schema-v3 artifact tree без зависимости от service package."""

    def __init__(self, *, artifact_root: Path, manifest: SpeechModelManifest) -> None:
        self._artifact_root = artifact_root.resolve()
        self._manifest = manifest

    def artifact_path(self, artifact: ModelArtifact) -> Path:
        target = (self._artifact_root / artifact.relative_path).resolve()
        if self._artifact_root not in target.parents:
            raise ValueError("model artifact path escapes artifact root")
        return target

    def verify_ready(self) -> bool:
        try:
            return all(
                self.artifact_path(artifact).is_dir()
                and artifact_tree_sha256(self.artifact_path(artifact)) == artifact.artifact_sha256
                for artifact in (self._manifest.asr, self._manifest.diarization)
            )
        except (OSError, ValueError):
            return False


def artifact_tree_sha256(path: Path) -> str:
    """Возвращает стабильный hash обычных файлов immutable artifact tree."""

    if not path.is_dir():
        raise ValueError("model artifact directory is unavailable")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("model artifact directory is empty")
    for item in files:
        if item.is_symlink():
            raise ValueError("model artifacts must not contain symlinks")
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
