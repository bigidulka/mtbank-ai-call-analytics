"""Ограниченные retries, global concurrency и circuit breaker для model gateway."""

from __future__ import annotations

import asyncio
import random as random_module
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from pydantic import model_validator

from mtbank_ai.agent_runtime.contracts import AgentFailureCode, AgentRuntimeError, ModelRequest, ModelResponse
from mtbank_ai.domain.base import PositiveFloat, PositiveInt, StrictFrozenModel
from mtbank_ai.observability import Telemetry


class RetryPolicy(StrictFrozenModel):
    max_attempts: PositiveInt = 3
    base_delay_seconds: PositiveFloat = 0.25
    max_delay_seconds: PositiveFloat = 2.0
    max_retry_after_seconds: PositiveFloat = 5.0

    @model_validator(mode="after")
    def validate_bounds(self) -> RetryPolicy:
        if self.max_attempts > 3:
            raise ValueError("max_attempts не может превышать 3")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds не может быть меньше base_delay_seconds")
        return self


class CircuitBreakerPolicy(StrictFrozenModel):
    failure_threshold: PositiveInt = 3
    recovery_seconds: PositiveFloat = 10.0


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse: ...


class CircuitBreaker:
    """Детерминированный process-local breaker с одним half-open probe."""

    def __init__(self, policy: CircuitBreakerPolicy, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._policy = policy
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False

    @property
    def state(self) -> CircuitState:
        return self._state

    def before_call(self) -> None:
        now = self._clock()
        if self._state is CircuitState.OPEN:
            assert self._opened_at is not None
            if now - self._opened_at < self._policy.recovery_seconds:
                raise AgentRuntimeError(AgentFailureCode.CIRCUIT_OPEN)
            self._state = CircuitState.HALF_OPEN
        if self._state is CircuitState.HALF_OPEN:
            if self._half_open_in_flight:
                raise AgentRuntimeError(AgentFailureCode.CIRCUIT_OPEN)
            self._half_open_in_flight = True

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None
        self._half_open_in_flight = False

    def record_failure(self) -> None:
        self._half_open_in_flight = False
        if self._state is CircuitState.HALF_OPEN:
            self._open()
            return
        self._failure_count += 1
        if self._failure_count >= self._policy.failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._failure_count = 0


class ResilientModelClient:
    """Единственная точка runtime retry; SDK automatic retries отключены provider-ом."""

    def __init__(
        self,
        client: ModelClient,
        *,
        max_concurrency: int,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
        random: Callable[[], float] = random_module.random,
        telemetry: Telemetry | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency должен быть положительным")
        self._client = client
        self._retry_policy = retry_policy or RetryPolicy()
        self._circuit_breaker = circuit_breaker or CircuitBreaker(CircuitBreakerPolicy())
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._now = now
        self._sleep = sleep
        self._random = random
        self._telemetry = telemetry or Telemetry()

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=_remaining_seconds(deadline_at, self._now))
        except TimeoutError:
            raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED) from None
        try:
            self._circuit_breaker.before_call()
            for attempt in range(self._retry_policy.max_attempts):
                timeout_seconds = _remaining_seconds(deadline_at, self._now)
                try:
                    response = await asyncio.wait_for(
                        self._client.complete(request, deadline_at=deadline_at),
                        timeout=timeout_seconds,
                    )
                except TimeoutError:
                    error_to_handle = AgentRuntimeError(AgentFailureCode.PROVIDER_TIMEOUT)
                except AgentRuntimeError as error:
                    error_to_handle = error
                else:
                    self._circuit_breaker.record_success()
                    return response

                if not _is_retryable(error_to_handle):
                    raise error_to_handle
                self._circuit_breaker.record_failure()
                if attempt + 1 >= self._retry_policy.max_attempts:
                    raise error_to_handle
                self._circuit_breaker.before_call()
                self._telemetry.metrics.increment("mtbank_agent_retries_total", reason=error_to_handle.code.value)
                delay = _retry_delay(error_to_handle, attempt, self._retry_policy, self._random)
                if delay >= _remaining_seconds(deadline_at, self._now):
                    raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED) from None
                try:
                    await asyncio.wait_for(self._sleep(delay), timeout=_remaining_seconds(deadline_at, self._now))
                except TimeoutError:
                    raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED) from None
        finally:
            self._semaphore.release()

        raise AssertionError("retry loop должен вернуть response или exception")


def _remaining_seconds(deadline_at: datetime, now: Callable[[], datetime]) -> float:
    remaining = (deadline_at - now()).total_seconds()
    if remaining <= 0:
        raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED)
    return remaining


def _is_retryable(error: AgentRuntimeError) -> bool:
    return error.code in {
        AgentFailureCode.PROVIDER_TRANSPORT,
        AgentFailureCode.PROVIDER_TIMEOUT,
        AgentFailureCode.PROVIDER_TOOL_USE_FAILED,
        AgentFailureCode.PROVIDER_RATE_LIMITED,
        AgentFailureCode.PROVIDER_SERVER,
    }


def _retry_delay(
    error: AgentRuntimeError,
    attempt: int,
    policy: RetryPolicy,
    random: Callable[[], float],
) -> float:
    retry_after = getattr(error, "retry_after_seconds", None)
    if isinstance(retry_after, (int, float)) and retry_after >= 0:
        return min(float(retry_after), policy.max_retry_after_seconds)
    capped = min(policy.base_delay_seconds * (2**attempt), policy.max_delay_seconds)
    jitter = 0.5 + 0.5 * min(max(random(), 0.0), 1.0)
    return capped * jitter
