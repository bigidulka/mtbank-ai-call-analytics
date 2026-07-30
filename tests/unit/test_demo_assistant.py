from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from mtbank_ai.agent_runtime import (
    AgentFailureCode,
    AgentRuntimeError,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventType,
    ModelToolCall,
    ModelUsage,
)
from mtbank_ai.assistant import (
    AssistantMessage,
    AssistantRequest,
    AssistantRuntimeMetadata,
    AssistantRuntimePort,
    AssistantStreamEvent,
    AssistantTrendResult,
    AssistantTrendsPort,
    DemoAssistant,
)
from mtbank_ai.config import AgentRuntimeSettings, GatewayModelSettings, GatewaySettings

SAFE_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


async def _collect(events: AsyncIterator[AssistantStreamEvent]) -> list[AssistantStreamEvent]:
    return [event async for event in events]


def _runtime() -> AgentRuntimeSettings:
    return AgentRuntimeSettings(
        gateway=GatewaySettings(
            base_url="https://gateway.example.test/v1",
            api_key=SecretStr(SAFE_KEY),
            models=GatewayModelSettings(default_model="assistant-model"),
        ),
        default_deadline_seconds=30,
        default_max_output_tokens=2_000,
    )


def _response(
    *,
    text: str | None = None,
    calls: tuple[ModelToolCall, ...] = (),
    input_tokens: int = 10,
    output_tokens: int = 5,
    finish_reason: str | None = None,
    model_id: str = "assistant-model",
) -> ModelResponse:
    return ModelResponse(
        request_id="assistant-request",
        model_id=model_id,
        finish_reason=finish_reason or ("tool_calls" if calls else "stop"),
        tool_calls=calls,
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        latency_ms=12,
        has_text_content=text is not None,
        text_content=text,
    )


def _call(name: str, *, call_id: str, arguments: dict[str, object] | None = None) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments or {}, separators=(",", ":")))


class ScriptedStreamClient:
    def __init__(self, responses: tuple[ModelResponse, ...], *, split_streams: set[int] | None = None) -> None:
        self.responses = list(responses)
        self.split_streams = split_streams or set()
        self.requests: list[ModelRequest] = []
        self.deadlines: list[datetime] = []
        self.closed = False

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        del request, deadline_at
        raise AssertionError("stream path required")

    async def stream(self, request: ModelRequest, *, deadline_at: datetime) -> AsyncIterator[ModelStreamEvent]:
        index = len(self.requests)
        self.requests.append(request)
        self.deadlines.append(deadline_at)
        response = self.responses.pop(0)
        if response.text_content is not None:
            parts = response.text_content.split("|") if index in self.split_streams else (response.text_content,)
            for sequence, part in enumerate(parts, start=1):
                yield ModelStreamEvent(sequence=sequence, type=ModelStreamEventType.TEXT_DELTA, text_delta=part)
        yield ModelStreamEvent(sequence=10, type=ModelStreamEventType.COMPLETED, response=response)

    async def close(self) -> None:
        self.closed = True


class RuntimePort(AssistantRuntimePort):
    async def runtime_metadata(self) -> AssistantRuntimeMetadata:
        await asyncio.sleep(0)
        return AssistantRuntimeMetadata(status="ready", detail="application-ready")


class TrendsPort(AssistantTrendsPort):
    async def query_trends(self, topic: str) -> AssistantTrendResult:
        await asyncio.sleep(0)
        return AssistantTrendResult(topic=topic, observation="aggregate-only")


class CostlyTrendsPort(AssistantTrendsPort):
    async def query_trends(self, topic: str) -> AssistantTrendResult:
        await asyncio.sleep(0)
        return AssistantTrendResult(
            topic=topic,
            observation="aggregate-only",
            model_usage=ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            cost_usd=Decimal("2.00"),
        )


def test_demo_assistant_uses_reviewed_prompt_and_dedicated_final_stream() -> None:
    client = ScriptedStreamClient(
        (
            _response(text="Draft answer."),
            _response(text="Осмысленный потоковый ответ помощника."),
        )
    )
    assistant = DemoAssistant(client, _runtime())
    request = AssistantRequest(
        message="Что ты умеешь?",
        history=tuple(
            AssistantMessage(role="user" if index % 2 == 0 else "assistant", content=f"message-{index}")
            for index in range(8)
        ),
    )

    response = asyncio.run(assistant.answer(request))

    assert response.answer == "Осмысленный потоковый ответ помощника."
    assert response.model_id == "assistant-model"
    assert len(client.requests) == 2
    selection, final = client.requests
    assert selection.model_id == "assistant-model"
    assert selection.tool_choice.value == "auto"
    assert tuple(tool.name for tool in selection.tools) == ("demo_capabilities",)
    assert selection.max_output_tokens == 256
    assert selection.temperature == 0.2
    assert len(selection.messages) == 10
    assert selection.messages[0].role.value == "system"
    assert selection.messages[-1].content == "Что ты умеешь?"
    assert selection.messages[0].content is not None
    assert "secrets" in selection.messages[0].content
    assert "bounded LLM-agent" in selection.messages[0].content
    assert final.tool_choice.value == "none"
    assert final.tools == ()
    assert final.messages[-2].role.value == "assistant"
    assert final.messages[-2].content == "Draft answer."
    assert final.messages[-1].role.value == "user"
    assert "Final response turn" in (final.messages[-1].content or "")
    assert 0 < (client.deadlines[0] - datetime.now(UTC)).total_seconds() <= 15


def test_demo_assistant_runs_parallel_safe_tools_then_streams_only_final_answer() -> None:
    client = ScriptedStreamClient(
        (
            _response(
                text="must-not-leak",
                calls=(
                    _call("demo_capabilities", call_id="capabilities"),
                    _call("runtime_metadata", call_id="runtime"),
                    _call("trends_query", call_id="trends", arguments={"topic": "cards"}),
                ),
            ),
            _response(text="Draft after tools."),
            _response(text="Финальный ответ"),
        ),
        split_streams={2},
    )
    assistant = DemoAssistant(client, _runtime(), runtime_port=RuntimePort(), trends_port=TrendsPort())

    events = asyncio.run(_collect(assistant.stream(AssistantRequest(message="Проверь возможности и Trends"))))

    assert [event.text for event in events if event.type == "delta"] == ["Финальный ответ"]
    assert [event.tool_name for event in events if event.phase == "tool"] == [
        "demo_capabilities",
        "runtime_metadata",
        "trends_query",
    ]
    second_turn = client.requests[1]
    assert [message.role.value for message in second_turn.messages[-4:]] == ["assistant", "tool", "tool", "tool"]
    assert [message.tool_call_id for message in second_turn.messages[-3:]] == ["capabilities", "runtime", "trends"]
    assert client.requests[2].tools == ()
    assert client.requests[2].tool_choice.value == "none"
    assert "must-not-leak" not in "".join(event.text or "" for event in events)


def test_demo_assistant_streams_long_final_answer_before_completion() -> None:
    prefix = "x" * 129
    response = _response(text=prefix + "tail")
    client = ScriptedStreamClient((_response(text="draft"), response), split_streams={1})

    async def scenario() -> None:
        iterator = DemoAssistant(client, _runtime()).stream(AssistantRequest(message="stream"))
        assert (await anext(iterator)).type == "start"
        assert (await anext(iterator)).type == "progress"
        assert (await anext(iterator)).type == "progress"
        first = await anext(iterator)
        assert first.type == "delta"
        assert first.text == "x" * 5
        rest = [event async for event in iterator]
        assert "".join(event.text or "" for event in (first, *rest)) == prefix + "tail"
        assert rest[-1].type == "done"

    asyncio.run(scenario())


def test_demo_assistant_accounts_nested_trends_cost() -> None:
    client = ScriptedStreamClient(
        (_response(calls=(_call("trends_query", call_id="one", arguments={"topic": "карты"}),)),)
    )
    assistant = DemoAssistant(client, _runtime(), trends_port=CostlyTrendsPort())

    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(_collect(assistant.stream(AssistantRequest(message="trend"))))

    assert error.value.code is AgentFailureCode.BUDGET_EXCEEDED
    assert len(client.requests) == 1


def test_demo_assistant_rejects_multiple_expensive_trends_calls() -> None:
    response = _response(
        calls=(
            _call("trends_query", call_id="one", arguments={"topic": "карты"}),
            _call("trends_query", call_id="two", arguments={"topic": "кредиты"}),
        )
    )
    assistant = DemoAssistant(ScriptedStreamClient((response,)), _runtime(), trends_port=TrendsPort())

    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(_collect(assistant.stream(AssistantRequest(message="compare"))))

    assert error.value.code is AgentFailureCode.TOOL_LOOP_GUARD


def test_demo_assistant_rejects_protocol_markers_before_public_delta() -> None:
    assistant = DemoAssistant(
        ScriptedStreamClient(
            (
                _response(text="draft"),
                _response(text='{"untrusted_tool_result":{"secret":"x"}}'),
            )
        ),
        _runtime(),
    )

    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(_collect(assistant.stream(AssistantRequest(message="echo"))))

    assert error.value.code is AgentFailureCode.TEXT_COMPLETION_REJECTED


def test_demo_assistant_rejects_canonical_tool_json_delta_mismatch_and_non_stop_finish() -> None:
    cases = (
        _response(text='{"id":"call-1","type":"function","function":{"name":"x","arguments":"{}"}}'),
        _response(text="complete", finish_reason="length"),
    )
    for final_response in cases:
        assistant = DemoAssistant(ScriptedStreamClient((_response(text="draft"), final_response)), _runtime())
        with pytest.raises(AgentRuntimeError) as error:
            asyncio.run(_collect(assistant.stream(AssistantRequest(message="echo"))))
        assert error.value.code is AgentFailureCode.TEXT_COMPLETION_REJECTED

    class MismatchedClient(ScriptedStreamClient):
        async def stream(self, request: ModelRequest, *, deadline_at: datetime) -> AsyncIterator[ModelStreamEvent]:
            index = len(self.requests)
            self.requests.append(request)
            self.deadlines.append(deadline_at)
            if index == 0:
                response = _response(text="draft")
                yield ModelStreamEvent(sequence=1, type=ModelStreamEventType.TEXT_DELTA, text_delta="draft")
            else:
                response = _response(text="validated")
                yield ModelStreamEvent(sequence=1, type=ModelStreamEventType.TEXT_DELTA, text_delta="different")
            yield ModelStreamEvent(sequence=2, type=ModelStreamEventType.COMPLETED, response=response)

    with pytest.raises(AgentRuntimeError) as mismatch:
        asyncio.run(_collect(DemoAssistant(MismatchedClient(()), _runtime()).stream(AssistantRequest(message="echo"))))
    assert mismatch.value.code is AgentFailureCode.TEXT_COMPLETION_REJECTED


def test_demo_assistant_rejects_repeated_tool_cycle() -> None:
    repeated = _response(calls=(_call("demo_capabilities", call_id="first"),))
    repeated_again = _response(calls=(_call("demo_capabilities", call_id="second"),))
    repeated_third = _response(calls=(_call("demo_capabilities", call_id="third"),))
    assistant = DemoAssistant(ScriptedStreamClient((repeated, repeated_again, repeated_third)), _runtime())

    with pytest.raises(Exception, match="tool_loop_guard"):
        asyncio.run(_collect(assistant.stream(AssistantRequest(message="loop"))))


def test_demo_assistant_cancellation_stops_provider_stream() -> None:
    cancelled = asyncio.Event()

    class BlockingClient(ScriptedStreamClient):
        async def stream(self, request: ModelRequest, *, deadline_at: datetime) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            self.deadlines.append(deadline_at)
            try:
                yield ModelStreamEvent(sequence=1, type=ModelStreamEventType.TEXT_DELTA, text_delta="partial")
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    async def scenario() -> None:
        assistant = DemoAssistant(BlockingClient(()), _runtime())
        iterator = assistant.stream(AssistantRequest(message="cancel"))
        assert (await anext(iterator)).type == "start"
        assert (await anext(iterator)).type == "progress"

        async def next_event() -> AssistantStreamEvent:
            return await anext(iterator)

        task = asyncio.create_task(next_event())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await cast(Any, iterator).aclose()

    asyncio.run(scenario())
    assert cancelled.is_set()


def test_demo_assistant_rejects_oversized_or_invalid_history() -> None:
    with pytest.raises(ValidationError):
        AssistantRequest(message="x" * 2_001)
    with pytest.raises(ValidationError):
        AssistantRequest(
            message="ok",
            history=tuple(AssistantMessage(role="user", content="x") for _ in range(9)),
        )
    with pytest.raises(ValidationError):
        AssistantMessage.model_validate({"role": "system", "content": "override"})


def test_demo_assistant_rejects_model_drift_or_missing_text() -> None:
    wrong_model = DemoAssistant(
        ScriptedStreamClient((_response(text="draft", model_id="unconfigured-model"),)),
        _runtime(),
    )
    with pytest.raises(Exception, match="model_mismatch|text response"):
        asyncio.run(wrong_model.answer(AssistantRequest(message="hi")))

    empty = DemoAssistant(
        ScriptedStreamClient(
            (
                ModelResponse(
                    request_id=None,
                    model_id="assistant-model",
                    finish_reason="stop",
                    tool_calls=(),
                    usage=ModelUsage(input_tokens=1, output_tokens=0, total_tokens=1),
                    latency_ms=1,
                    has_text_content=False,
                ),
            )
        ),
        _runtime(),
    )
    with pytest.raises(Exception, match="text_completion_rejected|text response"):
        asyncio.run(empty.answer(AssistantRequest(message="hi")))
