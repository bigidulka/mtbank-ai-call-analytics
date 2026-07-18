"""Append-only lifecycle event primitives."""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import model_validator

from mtbank_ai.domain.base import (
    JsonScalar,
    NonEmptyId,
    PositiveInt,
    Sha256,
    StrictFrozenModel,
    UtcDateTime,
)


class LifecycleEventType(StrEnum):
    RUN_STARTED = "RunStarted"
    SPEECH_STARTED = "SpeechStarted"
    SPEECH_COMPLETED = "SpeechCompleted"
    SPEECH_FAILED = "SpeechFailed"
    ROLE_RESOLUTION_STARTED = "RoleResolutionStarted"
    ROLE_RESOLUTION_COMPLETED = "RoleResolutionCompleted"
    AGENT_STARTED = "AgentStarted"
    MODEL_REQUESTED = "ModelRequested"
    MODEL_COMPLETED = "ModelCompleted"
    MODEL_FAILED = "ModelFailed"
    TOOL_PROPOSED = "ToolProposed"
    TOOL_VALIDATED = "ToolValidated"
    TOOL_ALLOWED = "ToolAllowed"
    TOOL_STARTED = "ToolStarted"
    TOOL_COMPLETED = "ToolCompleted"
    TOOL_FAILED = "ToolFailed"
    AGENT_OUTPUT_VALIDATED = "AgentOutputValidated"
    AGENT_OUTPUT_REJECTED = "AgentOutputRejected"
    AGGREGATION_COMPLETED = "AggregationCompleted"
    RUN_COMPLETED = "RunCompleted"
    RUN_FAILED = "RunFailed"


class RunStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EventAttribute(StrictFrozenModel):
    key: NonEmptyId
    value: JsonScalar


class RedactedPayload(StrictFrozenModel):
    """JSON object projection for the event payload JSONB column."""

    fields: tuple[EventAttribute, ...]

    @model_validator(mode="after")
    def require_unique_keys(self) -> Self:
        keys = tuple(attribute.key for attribute in self.fields)
        if len(set(keys)) != len(keys):
            raise ValueError("ключи redacted payload должны быть уникальны")
        return self


class RunEvent(StrictFrozenModel):
    run_id: UUID
    sequence: PositiveInt
    event_type: LifecycleEventType
    occurred_at: UtcDateTime
    component: NonEmptyId
    payload: RedactedPayload
    previous_hash: Sha256 | None
    current_hash: Sha256

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.sequence == 1 and self.previous_hash is not None:
            raise ValueError("первое событие не должно иметь previous_hash")
        if self.sequence > 1 and self.previous_hash is None:
            raise ValueError("событие после первого должно иметь previous_hash")
        return self
