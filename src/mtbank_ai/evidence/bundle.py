"""Минимальные ссылки на evidence и артефакты."""

from __future__ import annotations

from typing import Annotated, Self
from uuid import UUID

from pydantic import Field, model_validator

from mtbank_ai.domain.base import (
    MimeType,
    NonEmptyId,
    NonNegativeInt,
    Sha256,
    StrictFrozenModel,
    UtcDateTime,
)


class EvidenceReference(StrictFrozenModel):
    segment_ids: Annotated[tuple[UUID, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_segments(self) -> Self:
        if len(set(self.segment_ids)) != len(self.segment_ids):
            raise ValueError("evidence segment IDs должны быть уникальны")
        return self


class ArtifactDigest(StrictFrozenModel):
    name: NonEmptyId
    media_type: MimeType
    size_bytes: NonNegativeInt
    sha256: Sha256


class EvidenceBundleManifest(StrictFrozenModel):
    run_id: UUID
    envelope_sha256: Sha256
    events_sha256: Sha256
    artifacts: tuple[ArtifactDigest, ...]
    created_at: UtcDateTime
    schema_version: NonEmptyId = "1"

    @model_validator(mode="after")
    def require_unique_artifact_names(self) -> Self:
        names = tuple(artifact.name for artifact in self.artifacts)
        if len(set(names)) != len(names):
            raise ValueError("имена артефактов должны быть уникальны")
        return self
