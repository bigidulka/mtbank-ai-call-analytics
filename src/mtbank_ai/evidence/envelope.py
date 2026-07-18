"""Неизменяемый конверт воспроизводимости одного анализа."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from mtbank_ai.domain.base import (
    MimeType,
    NonEmptyId,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ReasoningEffort,
    Sha256,
    StrictFrozenModel,
    UtcDateTime,
)
from mtbank_ai.domain.provenance import ComponentRevision


class RunSource(StrEnum):
    OPENWEBUI = "openwebui"
    REST_FILE = "rest_file"
    REST_URL = "rest_url"
    WEBSOCKET = "websocket"
    EVAL = "eval"


class MediaDescriptor(StrictFrozenModel):
    sha256: Sha256
    mime_type: MimeType
    duration_seconds: PositiveFloat
    sample_rate_hz: PositiveInt
    channels: PositiveInt


class ModelBinding(StrictFrozenModel):
    agent_id: NonEmptyId
    provider_id: NonEmptyId
    model_id: NonEmptyId
    reasoning_effort: ReasoningEffort | None = None


class ProviderFingerprint(StrictFrozenModel):
    model_bindings: Annotated[tuple[ModelBinding, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def require_unique_agents(self) -> Self:
        agent_ids = tuple(binding.agent_id for binding in self.model_bindings)
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("agent_id в model bindings должны быть уникальны")
        return self


class RevisionSet(StrictFrozenModel):
    code_sha: NonEmptyId
    prompt_bundle_hash: Sha256
    taxonomy_version: NonEmptyId
    quality_rubric_version: NonEmptyId
    compliance_policy_version: NonEmptyId
    asr: ComponentRevision
    alignment: ComponentRevision
    diarization: ComponentRevision
    dataset_version: NonEmptyId | None = None
    eval_case_id: NonEmptyId | None = None

    @model_validator(mode="after")
    def require_dataset_for_eval_case(self) -> Self:
        if self.eval_case_id is not None and self.dataset_version is None:
            raise ValueError("eval_case_id требует dataset_version")
        return self


class RunBudget(StrictFrozenModel):
    deadline_at: UtcDateTime
    max_llm_turns: PositiveInt
    max_total_tokens: PositiveInt
    max_cost_usd: NonNegativeDecimal


class PrivacyPolicy(StrictFrozenModel):
    mode: NonEmptyId
    raw_audio_retention_seconds: NonNegativeInt
    evidence_retention_days: NonNegativeInt
    allow_full_content_evidence: bool


class RunEnvelope(StrictFrozenModel):
    schema_version: Literal["1"] = "1"
    run_id: UUID
    request_id: UUID
    correlation_id: UUID
    source: RunSource
    input_media: MediaDescriptor
    provider: ProviderFingerprint
    revisions: RevisionSet
    budget: RunBudget
    privacy: PrivacyPolicy
    created_at: UtcDateTime

    @model_validator(mode="after")
    def require_future_deadline(self) -> Self:
        if self.budget.deadline_at <= self.created_at:
            raise ValueError("deadline_at должен быть позже created_at")
        return self
