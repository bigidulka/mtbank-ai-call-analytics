from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest

from mtbank_ai.agent_runtime import (
    AgentFailureCode,
    CircuitBreaker,
    CircuitBreakerPolicy,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ProviderError,
    ResilientModelClient,
    RetryPolicy,
    ToolChoice,
    ToolRegistry,
    ToolSideEffect,
    ToolSpec,
)
from mtbank_ai.agent_runtime.contracts import ToolExecutionContext
from mtbank_ai.agent_runtime.tools import ToolExecutionError, ToolHandler, serialize_untrusted_observation
from mtbank_ai.domain.base import NonEmptyId, StrictFrozenModel

NOW = datetime(2026, 7, 16, tzinfo=UTC)


class NestedInput(StrictFrozenModel):
    value: NonEmptyId


class ToolInput(StrictFrozenModel):
    nested: NestedInput


class ToolOutput(StrictFrozenModel):
    result: NonEmptyId


async def _handler(arguments: ToolInput, context: ToolExecutionContext) -> ToolOutput:
    del arguments, context
    return ToolOutput(result="ok")


def _tool_registry() -> ToolRegistry:
    return ToolRegistry(
        (
            ToolSpec(
                "lookup",
                "Look up reviewed evidence.",
                ToolInput,
                ToolOutput,
                ToolSideEffect.READ_ONLY,
                1.0,
                _handler,
            ),
        )
    )


def _request() -> ModelRequest:
    return ModelRequest(
        model_id="configured-model",
        messages=(ModelMessage(role=MessageRole.SYSTEM, content="system"),),
        tools=(),
        tool_choice=ToolChoice.NONE,
        max_output_tokens=8,
    )


def _response() -> ModelResponse:
    return ModelResponse(
        request_id="request-id",
        model_id="configured-model",
        finish_reason="tool_calls",
        tool_calls=(),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        latency_ms=1,
        has_text_content=False,
    )


def test_registry_generates_recursive_strict_schema_and_validates_calls() -> None:
    registry = _tool_registry()
    schema: dict[str, Any] = registry.function_schemas(("lookup",))[0].parameters

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["nested"]
    assert schema["$defs"]["NestedInput"]["additionalProperties"] is False
    assert schema["$defs"]["NestedInput"]["required"] == ["value"]

    validated = registry.validate_calls(
        (ModelToolCall(id="call", name="lookup", arguments_json='{"nested":{"value":"x"}}'),)
    )
    assert validated[0].arguments == ToolInput(nested=NestedInput(value="x"))

    with pytest.raises(Exception) as duplicate:
        registry.validate_calls(
            (
                ModelToolCall(id="same", name="lookup", arguments_json='{"nested":{"value":"x"}}'),
                ModelToolCall(id="same", name="lookup", arguments_json='{"nested":{"value":"x"}}'),
            )
        )
    assert getattr(duplicate.value, "code") is AgentFailureCode.DUPLICATE_TOOL_CALL_ID


def test_observation_is_untrusted_bounded_and_not_an_event_payload() -> None:
    observation = serialize_untrusted_observation(
        tool_name="lookup",
        output=ToolOutput(result="safe"),
        max_bytes=128,
    )
    assert observation.untrusted_content == '{"untrusted_tool_result":{"result":"safe"}}'
    assert len(observation.observation_hash) == 64

    with pytest.raises(ToolExecutionError) as error:
        serialize_untrusted_observation(
            tool_name="lookup",
            output=ToolOutput(result="x" * 200),
            max_bytes=32,
        )
    assert error.value.code is AgentFailureCode.OBSERVATION_TOO_LARGE


def test_registry_rejects_sync_handlers_and_cancels_async_timeout() -> None:
    def sync_handler(arguments: ToolInput, context: ToolExecutionContext) -> ToolOutput:
        del arguments, context
        return ToolOutput(result="sync")

    with pytest.raises(TypeError, match="async"):
        ToolSpec(
            "sync_lookup",
            "Synchronous lookup.",
            ToolInput,
            ToolOutput,
            ToolSideEffect.READ_ONLY,
            1.0,
            cast(ToolHandler, sync_handler),
        )

    async def scenario() -> None:
        side_effects: list[str] = []
        cancelled = asyncio.Event()

        async def cancellable_handler(arguments: ToolInput, context: ToolExecutionContext) -> ToolOutput:
            del arguments, context
            side_effects.append("started")
            try:
                await asyncio.sleep(1)
                side_effects.append("completed")
                return ToolOutput(result="late")
            except asyncio.CancelledError:
                side_effects.append("cancelled")
                cancelled.set()
                raise

        registry = ToolRegistry(
            (
                ToolSpec(
                    "lookup",
                    "Look up reviewed evidence.",
                    ToolInput,
                    ToolOutput,
                    ToolSideEffect.READ_ONLY,
                    1.0,
                    cancellable_handler,
                ),
            )
        )
        call = registry.validate_calls(
            (ModelToolCall(id="call", name="lookup", arguments_json='{"nested":{"value":"x"}}'),)
        )[0]
        with pytest.raises(ToolExecutionError) as timeout:
            await registry.execute(
                call,
                ToolExecutionContext(
                    run_id=UUID("11111111-1111-4111-8111-111111111111"),
                    agent_id="agent",
                    deadline_at=NOW + timedelta(seconds=10),
                ),
                timeout_seconds=0.001,
                max_observation_bytes=128,
            )
        assert timeout.value.code is AgentFailureCode.TOOL_TIMEOUT
        await asyncio.wait_for(cancelled.wait(), timeout=0.1)
        await asyncio.sleep(0)
        assert side_effects == ["started", "cancelled"]

    asyncio.run(scenario())


def test_runtime_retry_honors_bounded_retry_after_and_hides_error_body() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
            del request, deadline_at
            self.calls += 1
            if self.calls == 1:
                raise ProviderError(
                    AgentFailureCode.PROVIDER_RATE_LIMITED,
                    retry_after_seconds=100.0,
                )
            return _response()

    waits: list[float] = []

    async def sleep(value: float) -> None:
        waits.append(value)

    client = ResilientModelClient(
        FlakyClient(),
        max_concurrency=1,
        retry_policy=RetryPolicy(max_attempts=2, max_retry_after_seconds=0.5),
        now=lambda: NOW,
        sleep=sleep,
        random=lambda: 0.0,
    )
    response = asyncio.run(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))

    assert response.request_id == "request-id"
    assert waits == [0.5]


def test_runtime_retries_transient_provider_tool_generation_failure() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
            del request, deadline_at
            self.calls += 1
            if self.calls == 1:
                raise ProviderError(AgentFailureCode.PROVIDER_TOOL_USE_FAILED)
            return _response()

    waits: list[float] = []

    async def sleep(value: float) -> None:
        waits.append(value)

    upstream = FlakyClient()
    client = ResilientModelClient(
        upstream,
        max_concurrency=1,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0.2),
        now=lambda: NOW,
        sleep=sleep,
        random=lambda: 0.0,
    )

    asyncio.run(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))

    assert upstream.calls == 2
    assert waits == [0.1]


def test_runtime_retry_uses_exponential_backoff_with_injected_jitter() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
            del request, deadline_at
            self.calls += 1
            if self.calls < 3:
                raise ProviderError(AgentFailureCode.PROVIDER_TRANSPORT)
            return _response()

    waits: list[float] = []

    async def sleep(value: float) -> None:
        waits.append(value)

    client = ResilientModelClient(
        FlakyClient(),
        max_concurrency=1,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.2, max_delay_seconds=1.0),
        now=lambda: NOW,
        sleep=sleep,
        random=lambda: 0.0,
    )
    asyncio.run(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))

    assert waits == [0.1, 0.2]


def test_circuit_breaker_opens_deterministically_after_failure() -> None:
    class FailingClient:
        async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
            del request, deadline_at
            raise ProviderError(AgentFailureCode.PROVIDER_SERVER)

    breaker = CircuitBreaker(CircuitBreakerPolicy(failure_threshold=1, recovery_seconds=10), clock=lambda: 1.0)
    client = ResilientModelClient(
        FailingClient(),
        max_concurrency=1,
        retry_policy=RetryPolicy(max_attempts=1),
        circuit_breaker=breaker,
        now=lambda: NOW,
    )

    with pytest.raises(ProviderError):
        asyncio.run(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))
    with pytest.raises(Exception) as error:
        asyncio.run(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))
    assert getattr(error.value, "code") is AgentFailureCode.CIRCUIT_OPEN


def test_global_semaphore_bounds_concurrent_gateway_calls() -> None:
    class BlockingClient:
        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
            del request, deadline_at
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.entered.set()
            await self.release.wait()
            self.active -= 1
            return _response()

    async def scenario() -> None:
        blocking = BlockingClient()
        client = ResilientModelClient(blocking, max_concurrency=1, now=lambda: NOW)
        first = asyncio.create_task(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))
        await blocking.entered.wait()
        second = asyncio.create_task(client.complete(_request(), deadline_at=NOW + timedelta(seconds=5)))
        await asyncio.sleep(0)
        assert blocking.maximum == 1
        assert blocking.active == 1
        blocking.release.set()
        await asyncio.gather(first, second)
        assert blocking.maximum == 1

    asyncio.run(scenario())
