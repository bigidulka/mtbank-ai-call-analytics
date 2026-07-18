from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from mtbank_ai.agent_runtime import (
    AgentBudget,
    AgentFailureCode,
    AgentRunContext,
    AgentRuntimeError,
    AgentSpec,
    BoundedAgentRuntime,
    InMemoryEventSink,
    MessageRole,
    ModelMessage,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    PromptReference,
    ToolRegistry,
    ToolSideEffect,
    ToolSpec,
)
from mtbank_ai.agent_runtime.events import EventRedactionError, LifecycleRecorder
from mtbank_ai.domain.base import NonEmptyId, StrictFrozenModel
from mtbank_ai.domain.events import LifecycleEventType

NOW = datetime(2026, 7, 16, tzinfo=UTC)
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


class ToolInput(StrictFrozenModel):
    value: NonEmptyId


class AgentOutput(StrictFrozenModel):
    value: NonEmptyId


class WrongOutput(StrictFrozenModel):
    unexpected: NonEmptyId


class ScriptedClient:
    def __init__(self, responses: Sequence[ModelResponse | BaseException]) -> None:
        self._responses = list(responses)
        self.requests = []

    async def complete(self, request: object, *, deadline_at: datetime) -> ModelResponse:
        del deadline_at
        self.requests.append(request)
        next_response = self._responses.pop(0)
        if isinstance(next_response, BaseException):
            raise next_response
        return next_response


def _call(name: str, *, call_id: str = "call-1", arguments: str = '{"value":"safe"}') -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=arguments)


def _response(
    *calls: ModelToolCall,
    input_tokens: int = 1,
    output_tokens: int = 1,
    text: bool = False,
    request_id: str = "provider-request",
) -> ModelResponse:
    return ModelResponse(
        request_id=request_id,
        model_id="configured-model",
        finish_reason="tool_calls" if calls else "stop",
        tool_calls=calls,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        latency_ms=3,
        has_text_content=text,
    )


def _context(*, deadline_at: datetime = NOW + timedelta(seconds=30)) -> AgentRunContext:
    return AgentRunContext(
        run_id=RUN_ID,
        run_version="run/v1",
        policy_version="policy/v1",
        created_at=NOW,
        deadline_at=deadline_at,
        messages=(ModelMessage(role=MessageRole.SYSTEM, content="prompt-private-content"),),
    )


def _spec(**changes: object) -> AgentSpec:
    values: dict[str, Any] = {
        "agent_id": "agent",
        "model_id": "configured-model",
        "model_version": "model/v1",
        "policy_version": "policy/v1",
        "prompt": PromptReference(
            prompt_id="agent",
            version="v1",
            content_hash="a" * 64,
            bundle_hash="b" * 64,
        ),
        "output_model": AgentOutput,
        "allowed_read_tools": ("retrieve",),
        "required_retrieval_tools": ("retrieve",),
        "terminal_submit_tool": "submit",
        "budget": AgentBudget(
            max_turns=3,
            max_input_tokens=10,
            max_output_tokens=10,
            max_cost_usd=Decimal("1"),
        ),
    }
    values.update(changes)
    return AgentSpec(**values)


def _registry(
    *,
    terminal_output_model: type[StrictFrozenModel] = AgentOutput,
    slow: bool = False,
    terminal_calls: list[str] | None = None,
) -> ToolRegistry:
    async def retrieve(arguments: ToolInput, context: object) -> AgentOutput:
        del arguments, context
        if slow:
            await asyncio.sleep(0.05)
        return AgentOutput(value="retrieved")

    async def submit(arguments: ToolInput, context: object) -> StrictFrozenModel:
        del context
        if terminal_calls is not None:
            terminal_calls.append(arguments.value)
        if terminal_output_model is WrongOutput:
            return WrongOutput(unexpected="wrong")
        return AgentOutput(value=arguments.value)

    async def forbidden(arguments: ToolInput, context: object) -> AgentOutput:
        del arguments, context
        return AgentOutput(value="forbidden")

    return ToolRegistry(
        (
            ToolSpec(
                "retrieve",
                "Retrieve reviewed evidence.",
                ToolInput,
                AgentOutput,
                ToolSideEffect.READ_ONLY,
                0.01,
                retrieve,
            ),
            ToolSpec(
                "submit",
                "Submit typed output.",
                ToolInput,
                terminal_output_model,
                ToolSideEffect.TERMINAL_SUBMIT,
                0.01,
                submit,
            ),
            ToolSpec(
                "forbidden", "Forbidden read tool.", ToolInput, AgentOutput, ToolSideEffect.READ_ONLY, 0.01, forbidden
            ),
        )
    )


def _runtime(
    client: ScriptedClient, registry: ToolRegistry, sink: InMemoryEventSink | None = None
) -> BoundedAgentRuntime:
    return BoundedAgentRuntime(client, registry, event_sink=sink, now=lambda: NOW)


def test_runtime_requires_retrieval_then_one_terminal_output_and_redacts_trajectory() -> None:
    remote_request_ids = ("provider-request-secret-one", "provider-request-secret-two")
    remote_tool_ids = ("tool-call-secret-one", "tool-call-secret-two")
    sink = InMemoryEventSink()
    client = ScriptedClient(
        (
            _response(
                _call("retrieve", call_id=remote_tool_ids[0], arguments='{"value":"raw transcript secret"}'),
                request_id=remote_request_ids[0],
            ),
            _response(
                _call("submit", call_id=remote_tool_ids[1], arguments='{"value":"approved"}'),
                request_id=remote_request_ids[1],
            ),
        )
    )

    result = asyncio.run(_runtime(client, _registry(), sink).run(_spec(reasoning_effort="high"), _context()))

    assert result.output == AgentOutput(value="approved")
    assert result.run_version == "run/v1"
    assert result.reasoning_effort == "high"
    assert result.evidence.reasoning_effort == "high"
    assert result.usage == ModelUsage(input_tokens=2, output_tokens=2, total_tokens=4)
    assert tuple(request.model_id for request in client.requests) == ("configured-model", "configured-model")
    assert {tool.name for tool in client.requests[0].tools} == {"retrieve"}
    assert {tool.name for tool in client.requests[1].tools} == {"submit"}
    assert client.requests[1].messages[-1].role is MessageRole.TOOL
    assert client.requests[1].messages[-1].tool_call_id == remote_tool_ids[0]
    assert "untrusted_tool_result" in client.requests[1].messages[-1].content
    assert result.trajectory[-1].event_type is LifecycleEventType.RUN_COMPLETED
    assert {record.model_call_id for record in result.trajectory if record.model_call_id} == {"model-1", "model-2"}
    assert {record.tool_call_id for record in result.trajectory if record.tool_call_id} == {"tool-1-1", "tool-2-1"}
    assert result.evidence.trajectory == result.trajectory
    rendered_evidence = result.evidence.model_dump_json()
    assert "approved" not in rendered_evidence
    assert "raw transcript secret" not in rendered_evidence
    assert all(remote_id not in rendered_evidence for remote_id in (*remote_request_ids, *remote_tool_ids))
    rendered_events = "\n".join(event.model_dump_json() for event in sink.events)
    assert "prompt-private-content" not in rendered_events
    assert "raw transcript secret" not in rendered_events
    assert "approved" not in rendered_events
    assert all(remote_id not in rendered_events for remote_id in (*remote_request_ids, *remote_tool_ids))
    assert tuple(event.sequence for event in sink.events) == tuple(range(1, len(sink.events) + 1))


def test_runtime_rejects_cross_turn_remote_tool_id_before_event_or_handler() -> None:
    remote_tool_id = "tool-call-secret-reused"
    remote_request_ids = ("provider-request-secret-one", "provider-request-secret-two")
    terminal_calls: list[str] = []
    sink = InMemoryEventSink()
    client = ScriptedClient(
        (
            _response(
                _call("retrieve", call_id=remote_tool_id),
                request_id=remote_request_ids[0],
            ),
            _response(
                _call("submit", call_id=remote_tool_id),
                request_id=remote_request_ids[1],
            ),
        )
    )

    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(_runtime(client, _registry(terminal_calls=terminal_calls), sink).run(_spec(), _context()))

    assert error.value.code is AgentFailureCode.DUPLICATE_TOOL_CALL_ID
    assert remote_tool_id not in str(error.value)
    assert terminal_calls == []
    assert sum(event.event_type is LifecycleEventType.TOOL_PROPOSED for event in sink.events) == 1
    rendered_events = "\n".join(event.model_dump_json() for event in sink.events)
    assert remote_tool_id not in rendered_events
    assert all(remote_id not in rendered_events for remote_id in remote_request_ids)


def test_lifecycle_recorder_rejects_raw_payload_keys_and_keeps_hash_chain() -> None:
    async def scenario() -> None:
        recorder = LifecycleRecorder(run_id=RUN_ID, now=lambda: NOW)
        first = await recorder.record(LifecycleEventType.AGENT_STARTED, payload={"attempt": 1})
        with pytest.raises(EventRedactionError):
            await recorder.record(LifecycleEventType.MODEL_FAILED, payload={"raw_response": "never-store"})
        second = await recorder.record(LifecycleEventType.RUN_FAILED, payload={"failure_code": "failed"})
        assert second.previous_hash == first.current_hash

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "expected"),
    (
        (_response(text=True), AgentFailureCode.TEXT_COMPLETION_REJECTED),
        (_response(_call("unknown")), AgentFailureCode.UNKNOWN_TOOL),
        (
            _response(_call("retrieve", call_id="same"), _call("retrieve", call_id="same")),
            AgentFailureCode.DUPLICATE_TOOL_CALL_ID,
        ),
        (_response(_call("retrieve", arguments="not-json")), AgentFailureCode.TOOL_ARGUMENTS_INVALID),
        (_response(_call("forbidden")), AgentFailureCode.TOOL_NOT_ALLOWED),
        (
            _response(_call("submit", call_id="one"), _call("retrieve", call_id="two")),
            AgentFailureCode.POST_TERMINAL_TOOL_CALL,
        ),
        (_response(_call("submit")), AgentFailureCode.REQUIRED_RETRIEVAL_MISSING),
    ),
)
def test_runtime_rejects_invalid_model_tool_paths(response: ModelResponse, expected: AgentFailureCode) -> None:
    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(_runtime(ScriptedClient((response,)), _registry()).run(_spec(), _context()))

    assert error.value.code is expected


def test_runtime_rejects_malformed_response_policy_drift_and_oversized_observation() -> None:
    with pytest.raises(AgentRuntimeError) as malformed_error:
        asyncio.run(_runtime(ScriptedClient((_response(),)), _registry()).run(_spec(), _context()))
    assert malformed_error.value.code is AgentFailureCode.MALFORMED_PROVIDER_RESPONSE

    mismatched_context = _context().model_copy(update={"policy_version": "policy/v2"})
    with pytest.raises(AgentRuntimeError) as policy_error:
        asyncio.run(_runtime(ScriptedClient(()), _registry()).run(_spec(), mismatched_context))
    assert policy_error.value.code is AgentFailureCode.POLICY_VERSION_MISMATCH

    tiny_observation_budget = AgentBudget(
        max_turns=3,
        max_input_tokens=10,
        max_output_tokens=10,
        max_cost_usd=Decimal("1"),
        max_observation_bytes=1,
    )
    with pytest.raises(AgentRuntimeError) as observation_error:
        asyncio.run(
            _runtime(ScriptedClient((_response(_call("retrieve")),)), _registry()).run(
                _spec(budget=tiny_observation_budget), _context()
            )
        )
    assert observation_error.value.code is AgentFailureCode.OBSERVATION_TOO_LARGE

    terminal_only = _spec(
        allowed_read_tools=(),
        required_retrieval_tools=(),
        budget=tiny_observation_budget,
    )
    terminal_result = asyncio.run(
        _runtime(ScriptedClient((_response(_call("submit")),)), _registry()).run(terminal_only, _context())
    )
    assert terminal_result.output == AgentOutput(value="safe")


def test_runtime_rejects_wrong_terminal_output_without_partial_success() -> None:
    client = ScriptedClient(
        (
            _response(_call("retrieve")),
            _response(_call("submit", call_id="submit-call")),
        )
    )
    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(_runtime(client, _registry(terminal_output_model=WrongOutput)).run(_spec(), _context()))

    assert error.value.code is AgentFailureCode.TERMINAL_SUBMIT_INVALID


def test_runtime_rejects_budget_deadline_and_missing_terminal() -> None:
    budget = AgentBudget(
        max_turns=3,
        max_input_tokens=1,
        max_output_tokens=10,
        max_cost_usd=Decimal("1"),
    )
    with pytest.raises(AgentRuntimeError) as budget_error:
        asyncio.run(
            _runtime(ScriptedClient((_response(_call("retrieve"), input_tokens=2),)), _registry()).run(
                _spec(budget=budget), _context()
            )
        )
    assert budget_error.value.code is AgentFailureCode.BUDGET_EXCEEDED

    no_terminal = ScriptedClient(
        tuple(_response(_call("retrieve", call_id=call_id)) for call_id in ("one", "two", "three"))
    )
    with pytest.raises(AgentRuntimeError) as terminal_error:
        asyncio.run(_runtime(no_terminal, _registry()).run(_spec(), _context()))
    assert terminal_error.value.code is AgentFailureCode.TERMINAL_SUBMIT_MISSING

    expired_context = AgentRunContext(
        run_id=RUN_ID,
        run_version="run/v1",
        policy_version="policy/v1",
        created_at=NOW - timedelta(seconds=1),
        deadline_at=NOW,
        messages=(ModelMessage(role=MessageRole.SYSTEM, content="prompt"),),
    )
    with pytest.raises(AgentRuntimeError) as deadline_error:
        asyncio.run(_runtime(ScriptedClient(()), _registry()).run(_spec(), expired_context))
    assert deadline_error.value.code is AgentFailureCode.DEADLINE_EXCEEDED


def test_runtime_rejects_tool_timeout_and_model_fallback() -> None:
    timeout_client = ScriptedClient((_response(_call("retrieve")),))
    with pytest.raises(AgentRuntimeError) as timeout_error:
        asyncio.run(_runtime(timeout_client, _registry(slow=True)).run(_spec(), _context()))
    assert timeout_error.value.code is AgentFailureCode.TOOL_TIMEOUT

    fallback = _response(_call("retrieve"))
    fallback = fallback.model_copy(update={"model_id": "unconfigured-model"})
    with pytest.raises(AgentRuntimeError) as model_error:
        asyncio.run(_runtime(ScriptedClient((fallback,)), _registry()).run(_spec(), _context()))
    assert model_error.value.code is AgentFailureCode.MODEL_MISMATCH


def test_agent_contracts_are_strict_and_limit_turns() -> None:
    with pytest.raises(ValidationError, match="max_turns"):
        _spec(
            budget=AgentBudget(
                max_turns=4,
                max_input_tokens=1,
                max_output_tokens=1,
                max_cost_usd=Decimal("0"),
            )
        )
    with pytest.raises(ValidationError):
        AgentBudget(
            max_turns="3",  # type: ignore[arg-type]
            max_input_tokens=1,
            max_output_tokens=1,
            max_cost_usd=Decimal("0"),
        )
