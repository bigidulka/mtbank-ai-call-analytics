"""Bounded autonomous text assistant with safe tool use and streamed final answers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol, cast
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from mtbank_ai.agent_runtime import (
    AgentFailureCode,
    AgentRuntimeError,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventType,
    ModelUsage,
    PromptRegistry,
    ToolChoice,
    ToolRegistry,
    ToolSideEffect,
    ToolSpec,
)
from mtbank_ai.agent_runtime.contracts import ToolExecutionContext
from mtbank_ai.agent_runtime.tools import ExecutedToolCall, ValidatedToolCall
from mtbank_ai.config import AgentRuntimeSettings
from mtbank_ai.domain.base import NonEmptyId, StrictFrozenModel

_PROMPT_ID = "demo_assistant"
_PROMPT_VERSION = "v1"
_PROMPT_ROOT = Path(__file__).resolve().parent / "agents"
_MAX_HISTORY_MESSAGES = 8
_MAX_MESSAGE_CHARS = 2_000
_MAX_OUTPUT_TOKENS = 500
_MAX_ANSWER_CHARS = 8_000
_MAX_TURNS = 6
_MAX_TOOL_CALLS = 16
_MAX_REPEATED_TOOL_CALLS = 2
_MAX_EXPENSIVE_TOOL_CALLS = 1
_MAX_OBSERVATION_BYTES = 8_000
_DEADLINE_SECONDS = 15.0


class AssistantMessage(StrictFrozenModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=_MAX_MESSAGE_CHARS)


class AssistantRequest(StrictFrozenModel):
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_CHARS)
    history: tuple[AssistantMessage, ...] = Field(default=(), max_length=_MAX_HISTORY_MESSAGES)

    @field_validator("history", mode="before")
    @classmethod
    def parse_history(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class AssistantResponse(StrictFrozenModel):
    answer: str = Field(min_length=1, max_length=8_000)
    model_id: str = Field(min_length=1, max_length=256)


class AssistantStreamEvent(StrictFrozenModel):
    """Public-safe event. Tool arguments, outputs and provider identifiers never leave runtime."""

    sequence: int = Field(ge=1)
    type: Literal["start", "progress", "delta", "done"]
    phase: Literal["model", "tool"] | None = None
    tool_name: NonEmptyId | None = None
    text: str | None = Field(default=None, min_length=1, max_length=8_000)

    @field_validator("tool_name")
    @classmethod
    def allow_only_public_tool_labels(cls, value: str | None) -> str | None:
        if value not in {None, "demo_capabilities", "runtime_metadata", "trends_query"}:
            raise ValueError("tool label не входит в public allowlist")
        return value

    @model_validator(mode="after")
    def validate_event_shape(self) -> AssistantStreamEvent:
        if self.type in {"start", "done"}:
            if self.phase is not None or self.tool_name is not None or self.text is not None:
                raise ValueError("start/done event не может содержать payload")
        elif self.type == "delta":
            if self.text is None or self.phase is not None or self.tool_name is not None:
                raise ValueError("delta event должен содержать только text")
        elif self.phase == "model":
            if self.tool_name is not None or self.text is not None:
                raise ValueError("model progress не может содержать tool или text")
        elif self.phase == "tool":
            if self.tool_name is None or self.text is not None:
                raise ValueError("tool progress требует public tool label")
        else:
            raise ValueError("progress event требует phase")
        return self


class AssistantRuntimeMetadata(StrictFrozenModel):
    status: Literal["ready", "unavailable"]
    detail: NonEmptyId


class AssistantTrendQuery(StrictFrozenModel):
    topic: NonEmptyId


class AssistantTrendResult(StrictFrozenModel):
    topic: NonEmptyId
    observation: str = Field(min_length=1, max_length=4_000)
    model_usage: ModelUsage | None = None
    cost_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))


class _EmptyToolInput(StrictFrozenModel):
    pass


class _DemoCapabilities(StrictFrozenModel):
    capabilities: tuple[str, ...]


class AssistantRuntimePort(Protocol):
    async def runtime_metadata(self) -> AssistantRuntimeMetadata: ...


class AssistantTrendsPort(Protocol):
    async def query_trends(self, topic: str) -> AssistantTrendResult: ...


class AssistantModelPort(Protocol):
    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse: ...

    async def close(self) -> None: ...


class DemoAssistant:
    """Server-owned bounded harness. Model chooses only static read-only tools."""

    def __init__(
        self,
        model_client: AssistantModelPort,
        runtime_settings: AgentRuntimeSettings,
        *,
        runtime_port: AssistantRuntimePort | None = None,
        trends_port: AssistantTrendsPort | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._model_client = model_client
        self._runtime_settings = runtime_settings
        self._runtime_port = runtime_port
        self._trends_port = trends_port
        self._prompt_registry = prompt_registry or PromptRegistry(_PROMPT_ROOT)

    async def answer(self, request: AssistantRequest) -> AssistantResponse:
        answer_parts: list[str] = []
        completed = False
        async for event in self.stream(request):
            if event.type == "delta":
                assert event.text is not None
                answer_parts.append(event.text)
            elif event.type == "done":
                completed = True
        answer = "".join(answer_parts).strip()
        if not answer or not completed:
            raise ValueError("assistant provider вернул некорректный text response")
        return AssistantResponse(answer=answer, model_id=self._runtime_settings.gateway.models.default_model)

    async def stream(self, request: AssistantRequest) -> AsyncIterator[AssistantStreamEvent]:
        registry = self._tool_registry()
        tools = registry.function_schemas(tuple(self._tool_names()))
        prompt = self._prompt_registry.load(
            _PROMPT_ID,
            _PROMPT_VERSION,
            policy_inputs={
                "deadline_seconds": min(_DEADLINE_SECONDS, self._runtime_settings.default_deadline_seconds),
                "history_messages": _MAX_HISTORY_MESSAGES,
                "message_chars": _MAX_MESSAGE_CHARS,
                "output_tokens": min(_MAX_OUTPUT_TOKENS, self._runtime_settings.default_max_output_tokens),
                "max_turns": min(_MAX_TURNS, self._runtime_settings.default_max_turns),
                "tools_enabled": bool(tools),
            },
            tool_schemas=tools,
        )
        model_id = self._runtime_settings.gateway.models.default_model
        deadline_at = datetime.now(UTC) + timedelta(
            seconds=min(_DEADLINE_SECONDS, self._runtime_settings.default_deadline_seconds)
        )
        messages = [ModelMessage(role=MessageRole.SYSTEM, content=prompt.text)]
        messages.extend(ModelMessage(role=MessageRole(item.role), content=item.content) for item in request.history)
        messages.append(ModelMessage(role=MessageRole.USER, content=request.message))
        sequence = 0
        tool_call_count = 0
        expensive_tool_call_count = 0
        repeated_calls: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost_usd = Decimal("0")

        sequence += 1
        yield AssistantStreamEvent(sequence=sequence, type="start")
        max_turns = min(_MAX_TURNS, self._runtime_settings.default_max_turns)
        for _turn in range(max_turns):
            max_output = min(_MAX_OUTPUT_TOKENS, self._runtime_settings.default_max_output_tokens)
            remaining_output = max_output - total_output_tokens
            if remaining_output <= 0:
                raise AgentRuntimeError(AgentFailureCode.BUDGET_EXCEEDED)
            self._require_remaining(deadline_at)
            sequence += 1
            yield AssistantStreamEvent(sequence=sequence, type="progress", phase="model")
            response, deltas = await self._stream_turn(
                ModelRequest(
                    model_id=model_id,
                    reasoning_effort=self._runtime_settings.gateway.models.default_reasoning_effort,
                    messages=tuple(messages),
                    tools=tools,
                    tool_choice=ToolChoice.AUTO if tools else ToolChoice.NONE,
                    max_output_tokens=remaining_output,
                    temperature=0.2,
                ),
                deadline_at=deadline_at,
                model_id=model_id,
            )
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            total_cost_usd += _usage_cost(response.usage, self._runtime_settings)
            if (
                total_input_tokens > self._runtime_settings.default_max_input_tokens
                or total_output_tokens > max_output
                or total_cost_usd > self._runtime_settings.default_max_cost_usd
            ):
                raise AgentRuntimeError(AgentFailureCode.BUDGET_EXCEEDED)
            if not response.tool_calls:
                if response.finish_reason != "stop" or response.text_content is None:
                    raise AgentRuntimeError(AgentFailureCode.TEXT_COMPLETION_REJECTED)
                visible = _validated_visible_answer(response.text_content, deltas)
                for delta in visible:
                    sequence += 1
                    yield AssistantStreamEvent(sequence=sequence, type="delta", text=delta)
                sequence += 1
                yield AssistantStreamEvent(sequence=sequence, type="done")
                return
            if response.finish_reason != "tool_calls":
                raise AgentRuntimeError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE)

            calls = registry.validate_calls(response.tool_calls)
            tool_call_count += len(calls)
            expensive_tool_call_count += sum(call.spec.name == "trends_query" for call in calls)
            if tool_call_count > _MAX_TOOL_CALLS or expensive_tool_call_count > _MAX_EXPENSIVE_TOOL_CALLS:
                raise AgentRuntimeError(AgentFailureCode.TOOL_LOOP_GUARD)
            for call in calls:
                signature = call.spec.name + "\n" + call.arguments.model_dump_json(by_alias=True, exclude_none=False)
                repeated_calls[signature] = repeated_calls.get(signature, 0) + 1
                if repeated_calls[signature] > _MAX_REPEATED_TOOL_CALLS:
                    raise AgentRuntimeError(AgentFailureCode.TOOL_LOOP_GUARD)
            messages.append(ModelMessage(role=MessageRole.ASSISTANT, tool_calls=response.tool_calls))
            for call in calls:
                sequence += 1
                yield AssistantStreamEvent(sequence=sequence, type="progress", phase="tool", tool_name=call.spec.name)
            results = await self._execute_tools(registry, calls, deadline_at=deadline_at)
            for call, result in zip(calls, results, strict=True):
                if result.observation is None:
                    raise AgentRuntimeError(AgentFailureCode.UNEXPECTED_RUNTIME_FAILURE)
                if isinstance(result.output, AssistantTrendResult) and result.output.model_usage is not None:
                    total_input_tokens += result.output.model_usage.input_tokens
                    total_output_tokens += result.output.model_usage.output_tokens
                    total_cost_usd += result.output.cost_usd
                    if (
                        total_input_tokens > self._runtime_settings.default_max_input_tokens
                        or total_output_tokens > max_output
                        or total_cost_usd > self._runtime_settings.default_max_cost_usd
                    ):
                        raise AgentRuntimeError(AgentFailureCode.BUDGET_EXCEEDED)
                messages.append(
                    ModelMessage(
                        role=MessageRole.TOOL,
                        content=result.observation.untrusted_content,
                        tool_call_id=call.call.id,
                    )
                )
        raise AgentRuntimeError(AgentFailureCode.TURN_LIMIT_EXCEEDED)

    async def _stream_turn(
        self,
        request: ModelRequest,
        *,
        deadline_at: datetime,
        model_id: str,
    ) -> tuple[ModelResponse, tuple[str, ...]]:
        stream = getattr(self._model_client, "stream", None)
        if not callable(stream):
            response = await self._model_client.complete(request, deadline_at=deadline_at)
            if response.model_id != model_id:
                raise AgentRuntimeError(AgentFailureCode.MODEL_MISMATCH)
            return response, (response.text_content,) if response.text_content is not None else ()
        typed_stream = cast(AsyncIterator[ModelStreamEvent], stream(request, deadline_at=deadline_at))
        deltas: list[str] = []
        completed: ModelResponse | None = None
        async for event in typed_stream:
            if event.type is ModelStreamEventType.TEXT_DELTA:
                assert event.text_delta is not None
                deltas.append(event.text_delta)
            elif event.type is ModelStreamEventType.COMPLETED:
                completed = event.response
            else:
                raise AgentRuntimeError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE)
        if completed is None:
            raise AgentRuntimeError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE)
        if completed.model_id != model_id:
            raise AgentRuntimeError(AgentFailureCode.MODEL_MISMATCH)
        return completed, tuple(deltas)

    async def _execute_tools(
        self,
        registry: ToolRegistry,
        calls: tuple[ValidatedToolCall, ...],
        *,
        deadline_at: datetime,
    ) -> tuple[ExecutedToolCall, ...]:
        async def execute(call: ValidatedToolCall):  # type: ignore[no-untyped-def]
            remaining = self._require_remaining(deadline_at)
            return await registry.execute(
                call,
                ToolExecutionContext(run_id=uuid4(), agent_id="demo_assistant", deadline_at=deadline_at),
                timeout_seconds=min(call.spec.timeout_seconds, remaining),
                max_observation_bytes=_MAX_OBSERVATION_BYTES,
            )

        tasks = [asyncio.create_task(execute(call)) for call in calls]
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _tool_registry(self) -> ToolRegistry:
        async def capabilities(_arguments: _EmptyToolInput, _context: ToolExecutionContext) -> _DemoCapabilities:
            return _DemoCapabilities(
                capabilities=(
                    "OpenWebUI model MTBank Attachment Probe",
                    "one WAV, MP3 or OGG attachment",
                    "transcript, roles, classification, quality, compliance, summary and action items",
                    "POST /analyze, POST /trends and WSS /ws/transcribe",
                    "Grafana call, quality, topic, latency, error and agent-token metrics",
                )
            )

        specs = [
            ToolSpec(
                "demo_capabilities",
                "Read reviewed public demo capabilities and supported verification paths.",
                _EmptyToolInput,
                _DemoCapabilities,
                ToolSideEffect.READ_ONLY,
                2.0,
                capabilities,
            )
        ]
        if self._runtime_port is not None:

            async def runtime_metadata(
                _arguments: _EmptyToolInput, _context: ToolExecutionContext
            ) -> AssistantRuntimeMetadata:
                assert self._runtime_port is not None
                return await self._runtime_port.runtime_metadata()

            specs.append(
                ToolSpec(
                    "runtime_metadata",
                    "Read safe current runtime readiness metadata. It cannot change runtime state.",
                    _EmptyToolInput,
                    AssistantRuntimeMetadata,
                    ToolSideEffect.READ_ONLY,
                    2.0,
                    runtime_metadata,
                )
            )
        if self._trends_port is not None:

            async def trends(arguments: AssistantTrendQuery, _context: ToolExecutionContext) -> AssistantTrendResult:
                assert self._trends_port is not None
                return await self._trends_port.query_trends(arguments.topic)

            specs.append(
                ToolSpec(
                    "trends_query",
                    "Read persisted aggregate Trends for one user-supplied topic without raw calls or PII.",
                    AssistantTrendQuery,
                    AssistantTrendResult,
                    ToolSideEffect.READ_ONLY,
                    3.0,
                    trends,
                )
            )
        return ToolRegistry(tuple(specs))

    def _tool_names(self) -> tuple[str, ...]:
        names = ["demo_capabilities"]
        if self._runtime_port is not None:
            names.append("runtime_metadata")
        if self._trends_port is not None:
            names.append("trends_query")
        return tuple(names)

    @staticmethod
    def _require_remaining(deadline_at: datetime) -> float:
        remaining = (deadline_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED)
        return remaining

    async def close(self) -> None:
        await self._model_client.close()


def _validated_visible_answer(text: str, deltas: tuple[str, ...]) -> tuple[str, ...]:
    normalized = text.strip()
    if not normalized or len(normalized) > _MAX_ANSWER_CHARS:
        raise AgentRuntimeError(AgentFailureCode.TEXT_COMPLETION_REJECTED)
    visible = deltas or (text,)
    emitted = "".join(visible).strip()
    if emitted != normalized or len(emitted) > _MAX_ANSWER_CHARS:
        raise AgentRuntimeError(AgentFailureCode.TEXT_COMPLETION_REJECTED)
    lowered = normalized.casefold()
    forbidden_markers = (
        "untrusted_tool_result",
        '"arguments_json"',
        '"tool_call_id"',
        '"tool_calls"',
        "<tool_call>",
        "<function_call>",
        "authorization: bearer",
    )
    if any(marker in lowered for marker in forbidden_markers) or _looks_like_tool_protocol_json(normalized):
        raise AgentRuntimeError(AgentFailureCode.TEXT_COMPLETION_REJECTED)
    return visible


def _looks_like_tool_protocol_json(text: str) -> bool:
    if not text.startswith(("{", "[")):
        return False
    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return _contains_tool_protocol(value)


def _contains_tool_protocol(value: object) -> bool:
    if isinstance(value, dict):
        keys = {str(key).casefold() for key in value}
        if {"name", "arguments"}.issubset(keys) or "tool_calls" in keys or "tool_call_id" in keys:
            return True
        if value.get("type") == "function" and isinstance(value.get("function"), dict):
            return True
        return any(_contains_tool_protocol(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_tool_protocol(child) for child in value)
    return False


def _usage_cost(usage: ModelUsage, settings: AgentRuntimeSettings) -> Decimal:
    models = settings.gateway.models
    return (
        Decimal(usage.input_tokens) * models.input_token_cost_usd
        + Decimal(usage.output_tokens) * models.output_token_cost_usd
    )
