"""Неизменяемые контракты bounded agent runtime."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from mtbank_ai.domain.base import (
    LongText,
    NonEmptyId,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveInt,
    ReasoningEffort,
    Sha256,
    StrictFrozenModel,
    UtcDateTime,
)
from mtbank_ai.domain.events import LifecycleEventType


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolChoice(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ToolSideEffect(StrEnum):
    READ_ONLY = "read_only"
    TERMINAL_SUBMIT = "terminal_submit"


class ToolCallStatus(StrEnum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    ALLOWED = "allowed"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentFailureCode(StrEnum):
    BUDGET_EXCEEDED = "budget_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    TURN_LIMIT_EXCEEDED = "turn_limit_exceeded"
    TEXT_COMPLETION_REJECTED = "text_completion_rejected"
    MALFORMED_PROVIDER_RESPONSE = "malformed_provider_response"
    UNKNOWN_TOOL = "unknown_tool"
    DUPLICATE_TOOL_CALL_ID = "duplicate_tool_call_id"
    TOOL_ARGUMENTS_INVALID = "tool_arguments_invalid"
    TOOL_NOT_ALLOWED = "tool_not_allowed"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    OBSERVATION_TOO_LARGE = "observation_too_large"
    REQUIRED_RETRIEVAL_MISSING = "required_retrieval_missing"
    TERMINAL_SUBMIT_MISSING = "terminal_submit_missing"
    TERMINAL_SUBMIT_INVALID = "terminal_submit_invalid"
    POST_TERMINAL_TOOL_CALL = "post_terminal_tool_call"
    POLICY_VERSION_MISMATCH = "policy_version_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    UNEXPECTED_RUNTIME_FAILURE = "unexpected_runtime_failure"
    CIRCUIT_OPEN = "circuit_open"
    PROVIDER_AUTHENTICATION = "provider_authentication"
    PROVIDER_PERMISSION = "provider_permission"
    PROVIDER_INVALID_REQUEST = "provider_invalid_request"
    PROVIDER_TOOL_USE_FAILED = "provider_tool_use_failed"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_TRANSPORT = "provider_transport"
    PROVIDER_SERVER = "provider_server"


class AgentRuntimeError(RuntimeError):
    """Контролируемое завершение агента без transcript или provider body."""

    def __init__(self, code: AgentFailureCode) -> None:
        self.code = code
        super().__init__(code.value)


class ModelToolCall(StrictFrozenModel):
    """Remote tool ID живёт только в краткоживущем protocol state."""

    id: NonEmptyId
    name: NonEmptyId
    arguments_json: Annotated[str, Field(min_length=1, max_length=65_536)]


class ModelMessage(StrictFrozenModel):
    role: MessageRole
    content: LongText | None = None
    tool_call_id: NonEmptyId | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.role in (MessageRole.SYSTEM, MessageRole.USER) and self.content is None:
            raise ValueError("system и user сообщения должны содержать content")
        if self.role is MessageRole.TOOL:
            if self.content is None or self.tool_call_id is None:
                raise ValueError("tool сообщение требует content и tool_call_id")
            if self.tool_calls:
                raise ValueError("tool сообщение не может содержать tool_calls")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id разрешён только для tool сообщения")
        if self.role is not MessageRole.ASSISTANT and self.tool_calls:
            raise ValueError("tool_calls разрешены только для assistant сообщения")
        if self.role is MessageRole.ASSISTANT and self.content is None and not self.tool_calls:
            raise ValueError("assistant сообщение требует content или tool_calls")
        return self


class FunctionToolSchema(StrictFrozenModel):
    name: NonEmptyId
    description: LongText
    parameters: dict[str, object]


class ModelRequest(StrictFrozenModel):
    model_id: NonEmptyId
    reasoning_effort: ReasoningEffort | None = None
    messages: Annotated[tuple[ModelMessage, ...], Field(min_length=1)]
    tools: tuple[FunctionToolSchema, ...]
    tool_choice: ToolChoice = ToolChoice.REQUIRED
    max_output_tokens: PositiveInt
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0

    @model_validator(mode="after")
    def require_unique_tools(self) -> Self:
        names = tuple(tool.name for tool in self.tools)
        if len(set(names)) != len(names):
            raise ValueError("имена tools должны быть уникальны")
        if self.tool_choice is ToolChoice.REQUIRED and not self.tools:
            raise ValueError("tool_choice required требует хотя бы один tool")
        return self


class ModelUsage(StrictFrozenModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    total_tokens: NonNegativeInt

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens должен равняться сумме input и output tokens")
        return self


class ModelResponse(StrictFrozenModel):
    request_id: NonEmptyId | None
    model_id: NonEmptyId
    finish_reason: NonEmptyId | None
    tool_calls: tuple[ModelToolCall, ...]
    usage: ModelUsage
    latency_ms: NonNegativeInt
    has_text_content: bool


class PromptReference(StrictFrozenModel):
    prompt_id: NonEmptyId
    version: NonEmptyId
    content_hash: Sha256
    bundle_hash: Sha256


class AgentBudget(StrictFrozenModel):
    max_turns: PositiveInt = 3
    max_input_tokens: PositiveInt
    max_output_tokens: PositiveInt
    max_cost_usd: NonNegativeDecimal
    input_token_cost_usd: NonNegativeDecimal = Decimal("0")
    output_token_cost_usd: NonNegativeDecimal = Decimal("0")
    max_observation_bytes: PositiveInt = 16_384

    @model_validator(mode="after")
    def validate_turn_bound(self) -> Self:
        if self.max_turns > 3:
            raise ValueError("max_turns bounded agent runtime не может превышать 3")
        if self.max_observation_bytes > 20_000:
            raise ValueError("max_observation_bytes не может превышать 20000")
        return self


class AgentSpec(StrictFrozenModel):
    agent_id: NonEmptyId
    model_id: NonEmptyId
    model_version: NonEmptyId
    reasoning_effort: ReasoningEffort | None = None
    policy_version: NonEmptyId
    prompt: PromptReference
    output_model: type[BaseModel]
    allowed_read_tools: tuple[NonEmptyId, ...]
    required_retrieval_tools: tuple[NonEmptyId, ...]
    terminal_submit_tool: NonEmptyId
    budget: AgentBudget

    @model_validator(mode="after")
    def validate_tools(self) -> Self:
        allowed = self.allowed_read_tools
        required = self.required_retrieval_tools
        if len(set(allowed)) != len(allowed):
            raise ValueError("allowed_read_tools должны быть уникальны")
        if len(set(required)) != len(required):
            raise ValueError("required_retrieval_tools должны быть уникальны")
        if self.terminal_submit_tool in allowed:
            raise ValueError("terminal submit tool не может быть read-only tool")
        if not set(required).issubset(allowed):
            raise ValueError("required retrieval tools должны входить в allowed_read_tools")
        return self


class AgentRunContext(StrictFrozenModel):
    run_id: UUID
    run_version: NonEmptyId
    policy_version: NonEmptyId
    created_at: UtcDateTime
    deadline_at: UtcDateTime
    messages: Annotated[tuple[ModelMessage, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_deadline(self) -> Self:
        if self.deadline_at <= self.created_at:
            raise ValueError("deadline_at должен быть позже created_at")
        return self


class SanitizedTrajectoryRecord(StrictFrozenModel):
    """Persistable trajectory только с run-local opaque IDs."""

    sequence: PositiveInt
    event_type: LifecycleEventType
    event_hash: Sha256
    model_id: NonEmptyId | None = None
    model_call_id: NonEmptyId | None = None
    tool_call_id: NonEmptyId | None = None
    tool_name: NonEmptyId | None = None
    status: ToolCallStatus | None = None
    usage: ModelUsage | None = None
    latency_ms: NonNegativeInt | None = None


class SanitizedAgentEvidence(StrictFrozenModel):
    """Разрешённый к persistence agent trace без output, prompt и observation body."""

    run_id: UUID
    run_version: NonEmptyId
    agent_id: NonEmptyId
    model_id: NonEmptyId
    model_version: NonEmptyId
    reasoning_effort: ReasoningEffort | None = None
    policy_version: NonEmptyId
    prompt: PromptReference
    usage: ModelUsage
    cost_usd: NonNegativeDecimal
    trajectory: tuple[SanitizedTrajectoryRecord, ...]


class AgentResult(StrictFrozenModel):
    run_id: UUID
    run_version: NonEmptyId
    agent_id: NonEmptyId
    model_id: NonEmptyId
    model_version: NonEmptyId
    reasoning_effort: ReasoningEffort | None = None
    policy_version: NonEmptyId
    prompt: PromptReference
    output: BaseModel
    usage: ModelUsage
    cost_usd: NonNegativeDecimal
    trajectory: tuple[SanitizedTrajectoryRecord, ...]
    evidence: SanitizedAgentEvidence


class ToolObservation(StrictFrozenModel):
    tool_name: NonEmptyId
    observation_hash: Sha256
    size_bytes: NonNegativeInt
    untrusted_content: Annotated[str, Field(min_length=1, max_length=65_536)]


class ToolExecutionContext(StrictFrozenModel):
    run_id: UUID
    agent_id: NonEmptyId
    deadline_at: UtcDateTime
