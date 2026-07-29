"""Bounded model → tools → terminal submit runtime без partial success."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, ValidationError

from mtbank_ai.agent_runtime.authorization import ToolAuthorizer
from mtbank_ai.agent_runtime.contracts import (
    AgentFailureCode,
    AgentResult,
    AgentRunContext,
    AgentRuntimeError,
    AgentSpec,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    SanitizedAgentEvidence,
    SanitizedTrajectoryRecord,
    ToolCallStatus,
    ToolChoice,
    ToolExecutionContext,
)
from mtbank_ai.agent_runtime.events import EventSink, LifecycleRecorder
from mtbank_ai.agent_runtime.retry import ModelClient
from mtbank_ai.agent_runtime.tools import ExecutedToolCall, ToolRegistry, ValidatedToolCall
from mtbank_ai.domain.events import LifecycleEventType, RunEvent
from mtbank_ai.observability import Telemetry

EventRecorder = Callable[..., Awaitable[RunEvent]]


class BoundedAgentRuntime:
    """Выполняет bounded autonomous model/tool loop и завершает только typed terminal output."""

    def __init__(
        self,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        *,
        event_sink: EventSink | None = None,
        authorizer: ToolAuthorizer | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        telemetry: Telemetry | None = None,
    ) -> None:
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._event_sink = event_sink
        self._authorizer = authorizer or ToolAuthorizer()
        self._now = now
        self._telemetry = telemetry or Telemetry()

    async def run(self, spec: AgentSpec, context: AgentRunContext) -> AgentResult:
        recorder = LifecycleRecorder(run_id=context.run_id, sink=self._event_sink, now=self._now)
        trajectory: list[SanitizedTrajectoryRecord] = []
        record_lock = asyncio.Lock()

        async def record(
            event_type: LifecycleEventType,
            *,
            payload: dict[str, str | int | float | bool | None] | None = None,
            model_id: str | None = None,
            model_call_id: str | None = None,
            tool_call_id: str | None = None,
            tool_name: str | None = None,
            status: ToolCallStatus | None = None,
            usage: ModelUsage | None = None,
            latency_ms: int | None = None,
        ) -> RunEvent:
            async with record_lock:
                event = await recorder.record(event_type, payload=payload)
                trajectory.append(
                    SanitizedTrajectoryRecord(
                        sequence=event.sequence,
                        event_type=event.event_type,
                        event_hash=event.current_hash,
                        model_id=model_id,
                        model_call_id=model_call_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        status=status,
                        usage=usage,
                        latency_ms=latency_ms,
                    )
                )
                return event

        try:
            return await self._run(spec, context, record, trajectory)
        except AgentRuntimeError as error:
            await record(
                LifecycleEventType.AGENT_OUTPUT_REJECTED,
                payload={"failure_code": error.code.value},
            )
            await record(
                LifecycleEventType.RUN_FAILED,
                payload={"failure_code": error.code.value},
            )
            raise
        except Exception:
            failure = AgentRuntimeError(AgentFailureCode.UNEXPECTED_RUNTIME_FAILURE)
            await record(
                LifecycleEventType.AGENT_OUTPUT_REJECTED,
                payload={"failure_code": failure.code.value},
            )
            await record(
                LifecycleEventType.RUN_FAILED,
                payload={"failure_code": failure.code.value},
            )
            raise failure from None

    async def _run(
        self,
        spec: AgentSpec,
        context: AgentRunContext,
        record: EventRecorder,
        trajectory: list[SanitizedTrajectoryRecord],
    ) -> AgentResult:
        if context.policy_version != spec.policy_version:
            raise AgentRuntimeError(AgentFailureCode.POLICY_VERSION_MISMATCH)
        self._require_remaining(context.deadline_at)

        terminal_spec = self._tool_registry.require(spec.terminal_submit_tool)
        if terminal_spec.side_effect.value != "terminal_submit":
            raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)

        await record(
            LifecycleEventType.AGENT_STARTED,
            payload={
                "agent_id": spec.agent_id,
                "model_id": spec.model_id,
                "model_version": spec.model_version,
                "policy_version": spec.policy_version,
                "prompt_bundle_hash": spec.prompt.bundle_hash,
            },
            model_id=spec.model_id,
        )

        messages = list(context.messages)
        retrieved_tools: set[str] = set()
        total_input_tokens = 0
        total_output_tokens = 0
        terminal_submitted = False
        seen_remote_tool_call_ids: set[str] = set()
        tool_call_count = 0
        repeated_calls: dict[str, int] = {}

        for turn in range(spec.budget.max_turns):
            model_call_id = f"model-{turn + 1}"
            remaining_output_tokens = spec.budget.max_output_tokens - total_output_tokens
            if remaining_output_tokens <= 0:
                raise AgentRuntimeError(AgentFailureCode.BUDGET_EXCEEDED)
            self._require_remaining(context.deadline_at)
            request = ModelRequest(
                model_id=spec.model_id,
                reasoning_effort=spec.reasoning_effort,
                messages=tuple(messages),
                tools=self._tool_registry.function_schemas((*spec.allowed_read_tools, spec.terminal_submit_tool)),
                tool_choice=ToolChoice.AUTO,
                max_output_tokens=remaining_output_tokens,
            )
            await record(
                LifecycleEventType.MODEL_REQUESTED,
                payload={"turn": turn + 1, "model_id": spec.model_id, "model_call_id": model_call_id},
                model_id=spec.model_id,
                model_call_id=model_call_id,
            )
            response = await self._complete_model(
                request,
                context.deadline_at,
                record,
                spec.model_id,
                model_call_id,
            )
            if response.model_id != spec.model_id:
                raise AgentRuntimeError(AgentFailureCode.MODEL_MISMATCH)

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            usage = ModelUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
            )
            await record(
                LifecycleEventType.MODEL_COMPLETED,
                payload={
                    "turn": turn + 1,
                    "model_id": response.model_id,
                    "model_call_id": model_call_id,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_ms": response.latency_ms,
                },
                model_id=response.model_id,
                model_call_id=model_call_id,
                usage=response.usage,
                latency_ms=response.latency_ms,
            )
            cost_usd = _cost(usage, spec)
            if (
                total_input_tokens > spec.budget.max_input_tokens
                or total_output_tokens > spec.budget.max_output_tokens
                or cost_usd > spec.budget.max_cost_usd
            ):
                raise AgentRuntimeError(AgentFailureCode.BUDGET_EXCEEDED)
            if response.has_text_content and not response.tool_calls:
                raise AgentRuntimeError(AgentFailureCode.TEXT_COMPLETION_REJECTED)
            if not response.tool_calls:
                raise AgentRuntimeError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE)

            validated_calls = self._tool_registry.validate_calls(
                response.tool_calls,
                seen_remote_call_ids=seen_remote_tool_call_ids,
            )
            tool_call_count += len(validated_calls)
            if tool_call_count > spec.budget.max_tool_calls:
                raise AgentRuntimeError(AgentFailureCode.TOOL_LOOP_GUARD)
            for validated in validated_calls:
                signature = _tool_signature(validated.spec.name, validated.arguments)
                repeated_calls[signature] = repeated_calls.get(signature, 0) + 1
                if repeated_calls[signature] > spec.budget.max_repeated_tool_calls:
                    raise AgentRuntimeError(AgentFailureCode.TOOL_LOOP_GUARD)
            local_tool_call_ids = tuple(f"tool-{turn + 1}-{index + 1}" for index in range(len(validated_calls)))
            for validated, tool_call_id in zip(validated_calls, local_tool_call_ids, strict=True):
                await record(
                    LifecycleEventType.TOOL_PROPOSED,
                    payload={"tool_call_id": tool_call_id, "tool_name": validated.spec.name},
                    tool_call_id=tool_call_id,
                    tool_name=validated.spec.name,
                    status=ToolCallStatus.PROPOSED,
                )
                await record(
                    LifecycleEventType.TOOL_VALIDATED,
                    payload={"tool_call_id": tool_call_id, "tool_name": validated.spec.name},
                    tool_call_id=tool_call_id,
                    tool_name=validated.spec.name,
                    status=ToolCallStatus.VALIDATED,
                )
            terminal_index = self._authorizer.authorize(
                spec=spec,
                calls=validated_calls,
                completed_retrieval_tools=retrieved_tools,
                terminal_submitted=terminal_submitted,
            )
            for validated, tool_call_id in zip(validated_calls, local_tool_call_ids, strict=True):
                await record(
                    LifecycleEventType.TOOL_ALLOWED,
                    payload={"tool_call_id": tool_call_id, "tool_name": validated.spec.name},
                    tool_call_id=tool_call_id,
                    tool_name=validated.spec.name,
                    status=ToolCallStatus.ALLOWED,
                )

            messages.append(ModelMessage(role=MessageRole.ASSISTANT, tool_calls=response.tool_calls))
            read_calls = tuple(
                (index, validated) for index, validated in enumerate(validated_calls) if index != terminal_index
            )
            read_results = await self._execute_read_calls(
                read_calls,
                local_tool_call_ids=local_tool_call_ids,
                context=context,
                spec=spec,
                record=record,
            )
            for index, validated in read_calls:
                executed = read_results[index]
                if validated.spec.name in spec.required_retrieval_tools:
                    retrieved_tools.add(validated.spec.name)
                if executed.observation is None:
                    raise AgentRuntimeError(AgentFailureCode.UNEXPECTED_RUNTIME_FAILURE)
                messages.append(
                    ModelMessage(
                        role=MessageRole.TOOL,
                        content=executed.observation.untrusted_content,
                        tool_call_id=validated.call.id,
                    )
                )

            if terminal_index is not None:
                validated = validated_calls[terminal_index]
                tool_call_id = local_tool_call_ids[terminal_index]
                executed = await self._execute_tool(
                    validated,
                    tool_call_id=tool_call_id,
                    context=context,
                    spec=spec,
                    record=record,
                    include_observation=False,
                )
                terminal_submitted = True
                output = _validate_terminal_output(spec, executed.output)
                await record(
                    LifecycleEventType.AGENT_OUTPUT_VALIDATED,
                    payload={"agent_id": spec.agent_id, "terminal_tool": validated.spec.name},
                    tool_call_id=tool_call_id,
                    tool_name=validated.spec.name,
                )
                await record(
                    LifecycleEventType.RUN_COMPLETED,
                    payload={"agent_id": spec.agent_id, "model_id": spec.model_id},
                    model_id=spec.model_id,
                )
                return _agent_result(
                    spec=spec,
                    context=context,
                    output=output,
                    usage=usage,
                    cost_usd=cost_usd,
                    trajectory=tuple(trajectory),
                )

        if not set(spec.required_retrieval_tools).issubset(retrieved_tools):
            raise AgentRuntimeError(AgentFailureCode.REQUIRED_RETRIEVAL_MISSING)
        raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_MISSING)

    async def _execute_read_calls(
        self,
        calls: tuple[tuple[int, ValidatedToolCall], ...],
        *,
        local_tool_call_ids: tuple[str, ...],
        context: AgentRunContext,
        spec: AgentSpec,
        record: EventRecorder,
    ) -> dict[int, ExecutedToolCall]:
        tasks = {
            index: asyncio.create_task(
                self._execute_tool(
                    validated,
                    tool_call_id=local_tool_call_ids[index],
                    context=context,
                    spec=spec,
                    record=record,
                    include_observation=True,
                )
            )
            for index, validated in calls
        }
        if not tasks:
            return {}
        try:
            results = await asyncio.gather(*tasks.values())
        except BaseException:
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise
        return dict(zip(tasks, results, strict=True))

    async def _execute_tool(
        self,
        validated: ValidatedToolCall,
        *,
        tool_call_id: str,
        context: AgentRunContext,
        spec: AgentSpec,
        record: EventRecorder,
        include_observation: bool,
    ) -> ExecutedToolCall:
        await record(
            LifecycleEventType.TOOL_STARTED,
            payload={"tool_call_id": tool_call_id, "tool_name": validated.spec.name},
            tool_call_id=tool_call_id,
            tool_name=validated.spec.name,
        )
        try:
            executed = await self._tool_registry.execute(
                validated,
                ToolExecutionContext(
                    run_id=context.run_id,
                    agent_id=spec.agent_id,
                    deadline_at=context.deadline_at,
                ),
                timeout_seconds=min(validated.spec.timeout_seconds, self._require_remaining(context.deadline_at)),
                max_observation_bytes=spec.budget.max_observation_bytes,
                include_observation=include_observation,
            )
        except AgentRuntimeError:
            await record(
                LifecycleEventType.TOOL_FAILED,
                payload={"tool_call_id": tool_call_id, "tool_name": validated.spec.name},
                tool_call_id=tool_call_id,
                tool_name=validated.spec.name,
                status=ToolCallStatus.FAILED,
            )
            raise
        payload: dict[str, str | int | float | bool | None] = {
            "tool_call_id": tool_call_id,
            "tool_name": validated.spec.name,
        }
        if executed.observation is not None:
            payload["observation_hash"] = executed.observation.observation_hash
            payload["observation_bytes"] = executed.observation.size_bytes
        await record(
            LifecycleEventType.TOOL_COMPLETED,
            payload=payload,
            tool_call_id=tool_call_id,
            tool_name=validated.spec.name,
            status=ToolCallStatus.COMPLETED,
        )
        return executed

    async def _complete_model(
        self,
        request: ModelRequest,
        deadline_at: datetime,
        record: EventRecorder,
        model_id: str,
        model_call_id: str,
    ) -> ModelResponse:
        try:
            with self._telemetry.span("agent.model_turn", model_id=model_id):
                response = await asyncio.wait_for(
                    self._model_client.complete(request, deadline_at=deadline_at),
                    timeout=self._require_remaining(deadline_at),
                )
            self._telemetry.metrics.increment("mtbank_agent_model_calls_total", model_id=model_id, status="ok")
            self._telemetry.metrics.increment(
                "mtbank_agent_tokens_total", model_id=model_id, direction="input", value=response.usage.input_tokens
            )
            self._telemetry.metrics.increment(
                "mtbank_agent_tokens_total", model_id=model_id, direction="output", value=response.usage.output_tokens
            )
            return response
        except TimeoutError:
            await record(
                LifecycleEventType.MODEL_FAILED,
                payload={
                    "failure_code": AgentFailureCode.DEADLINE_EXCEEDED.value,
                    "model_id": model_id,
                    "model_call_id": model_call_id,
                },
                model_id=model_id,
                model_call_id=model_call_id,
            )
            raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED) from None
        except AgentRuntimeError as error:
            await record(
                LifecycleEventType.MODEL_FAILED,
                payload={
                    "failure_code": error.code.value,
                    "model_id": model_id,
                    "model_call_id": model_call_id,
                },
                model_id=model_id,
                model_call_id=model_call_id,
            )
            raise

    def _require_remaining(self, deadline_at: datetime) -> float:
        remaining = (deadline_at - self._now()).total_seconds()
        if remaining <= 0:
            raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED)
        return remaining


def _tool_signature(name: str, arguments: BaseModel) -> str:
    return name + "\n" + arguments.model_dump_json(by_alias=True, exclude_none=False)


def _agent_result(
    *,
    spec: AgentSpec,
    context: AgentRunContext,
    output: BaseModel,
    usage: ModelUsage,
    cost_usd: Decimal,
    trajectory: tuple[SanitizedTrajectoryRecord, ...],
) -> AgentResult:
    evidence = SanitizedAgentEvidence(
        run_id=context.run_id,
        run_version=context.run_version,
        agent_id=spec.agent_id,
        model_id=spec.model_id,
        model_version=spec.model_version,
        reasoning_effort=spec.reasoning_effort,
        policy_version=spec.policy_version,
        prompt=spec.prompt,
        usage=usage,
        cost_usd=cost_usd,
        trajectory=trajectory,
    )
    return AgentResult(
        run_id=context.run_id,
        run_version=context.run_version,
        agent_id=spec.agent_id,
        model_id=spec.model_id,
        model_version=spec.model_version,
        reasoning_effort=spec.reasoning_effort,
        policy_version=spec.policy_version,
        prompt=spec.prompt,
        output=output,
        usage=usage,
        cost_usd=cost_usd,
        trajectory=trajectory,
        evidence=evidence,
    )


def _cost(usage: ModelUsage, spec: AgentSpec) -> Decimal:
    return (
        Decimal(usage.input_tokens) * spec.budget.input_token_cost_usd
        + Decimal(usage.output_tokens) * spec.budget.output_token_cost_usd
    )


def _validate_terminal_output(spec: AgentSpec, output: BaseModel) -> BaseModel:
    try:
        return spec.output_model.model_validate(output.model_dump(), strict=True)
    except (TypeError, ValidationError):
        raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID) from None
