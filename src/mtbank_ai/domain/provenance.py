"""Доверенные версии компонентов речевой обработки."""

from __future__ import annotations

from mtbank_ai.domain.base import NonEmptyId, Sha256, StrictFrozenModel


class ComponentRevision(StrictFrozenModel):
    """Воспроизводимая идентификация одного speech-компонента."""

    package: NonEmptyId
    package_version: NonEmptyId
    model_id: NonEmptyId
    model_revision: NonEmptyId
    artifact_sha256: Sha256 | None = None
