"""Официальный OpenAI SDK adapter для единственного cloud gateway boundary."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from mtbank_ai.agent_runtime.capabilities import StreamingProbeResult
from mtbank_ai.agent_runtime.contracts import (
    AgentFailureCode,
    AgentRuntimeError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from mtbank_ai.agent_runtime.retry import CircuitBreaker, CircuitBreakerPolicy, ResilientModelClient, RetryPolicy
from mtbank_ai.config import GatewaySettings
from mtbank_ai.observability import Telemetry


def _trusted_local_http_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False)


class ProviderError(AgentRuntimeError):
    """Sanitized provider failure: никогда не содержит response body или secret."""

    def __init__(
        self,
        code: AgentFailureCode,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(code)


class OpenAICompatibleProvider:
    """Chat Completions-only adapter с выключенными SDK retries."""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        client: Any | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        timeout = httpx.Timeout(settings.request_timeout_seconds, connect=settings.connect_timeout_seconds)
        if client is not None:
            self._client: Any = client
        elif settings.transport_mode == "trusted_local_http":
            self._client = AsyncOpenAI(
                api_key=settings.api_key.get_secret_value(),
                base_url=settings.base_url,
                timeout=timeout,
                max_retries=0,
                http_client=_trusted_local_http_client(timeout),
            )
        else:
            self._client = AsyncOpenAI(
                api_key=settings.api_key.get_secret_value(),
                base_url=settings.base_url,
                timeout=timeout,
                max_retries=0,
            )
        self._now = now
        self._monotonic = monotonic

    async def close(self) -> None:
        await self._client.close()

    async def probe_streaming(self, request: ModelRequest, *, deadline_at: datetime) -> StreamingProbeResult:
        """Проверяет usage в ограниченном stream и явное закрытие второго stream."""

        timeout_seconds = (deadline_at - self._now()).total_seconds()
        if timeout_seconds <= 0:
            raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED)
        try:
            return await asyncio.wait_for(self._probe_streaming(request), timeout=timeout_seconds)
        except TimeoutError:
            raise ProviderError(AgentFailureCode.PROVIDER_TIMEOUT) from None
        except openai.APITimeoutError:
            raise ProviderError(AgentFailureCode.PROVIDER_TIMEOUT) from None
        except openai.APIConnectionError:
            raise ProviderError(AgentFailureCode.PROVIDER_TRANSPORT) from None
        except openai.RateLimitError as error:
            raise _status_error(AgentFailureCode.PROVIDER_RATE_LIMITED, error) from None
        except openai.AuthenticationError as error:
            raise _status_error(AgentFailureCode.PROVIDER_AUTHENTICATION, error) from None
        except openai.PermissionDeniedError as error:
            raise _status_error(AgentFailureCode.PROVIDER_PERMISSION, error) from None
        except openai.BadRequestError as error:
            raise _bad_request_error(error) from None
        except openai.APIStatusError as error:
            raise _map_status_error(error) from None
        except openai.APIError:
            raise ProviderError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE) from None
        except Exception:
            raise ProviderError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE) from None

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        timeout_seconds = (deadline_at - self._now()).total_seconds()
        if timeout_seconds <= 0:
            raise AgentRuntimeError(AgentFailureCode.DEADLINE_EXCEEDED)
        started = self._monotonic()
        kwargs: dict[str, object] = {
            "model": request.model_id,
            "messages": [_message_payload(message) for message in request.messages],
            "tools": [_tool_payload(tool) for tool in request.tools],
            "tool_choice": request.tool_choice.value,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(**kwargs),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            raise ProviderError(AgentFailureCode.PROVIDER_TIMEOUT) from None
        except openai.APITimeoutError:
            raise ProviderError(AgentFailureCode.PROVIDER_TIMEOUT) from None
        except openai.APIConnectionError:
            raise ProviderError(AgentFailureCode.PROVIDER_TRANSPORT) from None
        except openai.RateLimitError as error:
            raise _status_error(AgentFailureCode.PROVIDER_RATE_LIMITED, error) from None
        except openai.AuthenticationError as error:
            raise _status_error(AgentFailureCode.PROVIDER_AUTHENTICATION, error) from None
        except openai.PermissionDeniedError as error:
            raise _status_error(AgentFailureCode.PROVIDER_PERMISSION, error) from None
        except openai.BadRequestError as error:
            raise _bad_request_error(error) from None
        except openai.APIStatusError as error:
            raise _map_status_error(error) from None
        except openai.APIError:
            raise ProviderError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE) from None
        except Exception:
            raise ProviderError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE) from None

        latency_ms = max(0, int((self._monotonic() - started) * 1_000))
        try:
            choice = completion.choices[0]
            message = choice.message
            raw_calls = message.tool_calls or ()
            tool_calls = tuple(
                ModelToolCall(
                    id=_require_remote_id(getattr(call, "id", None)),
                    name=call.function.name,
                    arguments_json=call.function.arguments,
                )
                for call in raw_calls
            )
            usage = completion.usage
            if usage is None:
                raise ValueError("usage отсутствует")
            response_usage = ModelUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            model_id = completion.model
            if not model_id:
                raise ValueError("model отсутствует")
        except (AttributeError, IndexError, TypeError, ValueError):
            raise AgentRuntimeError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE) from None

        try:
            return ModelResponse(
                request_id=_optional_remote_id(getattr(completion, "_request_id", None)),
                model_id=model_id,
                finish_reason=choice.finish_reason,
                tool_calls=tool_calls,
                usage=response_usage,
                latency_ms=latency_ms,
                has_text_content=bool(message.content and message.content.strip()),
            )
        except (TypeError, ValueError):
            raise AgentRuntimeError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE) from None

    async def _probe_streaming(self, request: ModelRequest) -> StreamingProbeResult:
        kwargs = _stream_kwargs(request)
        usage: ModelUsage | None = None
        streamed_model_id: str | None = None
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            streamed_model_id = _stream_model_id(streamed_model_id, chunk)
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage is not None:
                usage = ModelUsage(
                    input_tokens=raw_usage.prompt_tokens,
                    output_tokens=raw_usage.completion_tokens,
                    total_tokens=raw_usage.total_tokens,
                )
        cancelled = False
        cancellation_stream = await self._client.chat.completions.create(**kwargs)
        try:
            async for chunk in cancellation_stream:
                streamed_model_id = _stream_model_id(streamed_model_id, chunk)
                cancelled = True
                break
        finally:
            await _close_stream(cancellation_stream)
        return StreamingProbeResult(
            model_id=streamed_model_id,
            cancelled=cancelled,
            usage=usage,
            limit_enforced=usage is not None and usage.output_tokens <= request.max_output_tokens,
        )


class ConfiguredOpenAICompatibleGateway:
    """Соединяет typed gateway settings с единственными runtime controls."""

    def __init__(
        self,
        settings: GatewaySettings,
        *,
        client: Any | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._provider = OpenAICompatibleProvider(settings, client=client)
        self._runtime = ResilientModelClient(
            self._provider,
            max_concurrency=settings.max_concurrency,
            retry_policy=RetryPolicy(
                max_attempts=settings.retry_max_attempts,
                base_delay_seconds=settings.retry_base_delay_seconds,
                max_delay_seconds=settings.retry_max_delay_seconds,
                max_retry_after_seconds=settings.retry_max_retry_after_seconds,
            ),
            circuit_breaker=CircuitBreaker(
                CircuitBreakerPolicy(
                    failure_threshold=settings.circuit_failure_threshold,
                    recovery_seconds=settings.circuit_recovery_seconds,
                )
            ),
            telemetry=telemetry,
        )

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._runtime.circuit_breaker

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        return await self._runtime.complete(request, deadline_at=deadline_at)

    async def probe_streaming(self, request: ModelRequest, *, deadline_at: datetime) -> StreamingProbeResult:
        return await self._provider.probe_streaming(request, deadline_at=deadline_at)

    async def close(self) -> None:
        await self._provider.close()


def _optional_remote_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("provider ID должен быть строкой")
    normalized = value.strip()
    if not normalized or len(normalized) > 256:
        raise ValueError("provider ID должен быть непустым и ограниченным")
    return normalized


def _require_remote_id(value: object) -> str:
    remote_id = _optional_remote_id(value)
    if remote_id is None:
        raise ValueError("provider tool ID обязателен")
    return remote_id


def _stream_model_id(current: str | None, chunk: object) -> str | None:
    candidate = _optional_remote_id(getattr(chunk, "model", None))
    if current is not None and candidate is not None and current != candidate:
        raise ValueError("stream provider model changed")
    return candidate or current


def _message_payload(message: ModelMessage) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role.value}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in message.tool_calls
        ]
    return payload


def _tool_payload(tool: object) -> dict[str, object]:
    name = getattr(tool, "name")
    description = getattr(tool, "description")
    parameters = getattr(tool, "parameters")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _stream_kwargs(request: ModelRequest) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": request.model_id,
        "messages": [_message_payload(message) for message in request.messages],
        "tools": [_tool_payload(tool) for tool in request.tools],
        "tool_choice": request.tool_choice.value,
        "max_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.reasoning_effort is not None:
        kwargs["reasoning_effort"] = request.reasoning_effort
    return kwargs


async def _close_stream(stream: object) -> None:
    for method_name in ("aclose", "close"):
        method = getattr(stream, method_name, None)
        if callable(method):
            result = method()
            if inspect.isawaitable(result):
                await result
            return


def _bad_request_error(error: openai.BadRequestError) -> ProviderError:
    body = error.body
    if isinstance(body, Mapping) and body.get("code") == "tool_use_failed":
        return _status_error(AgentFailureCode.PROVIDER_TOOL_USE_FAILED, error)
    return _status_error(AgentFailureCode.PROVIDER_INVALID_REQUEST, error)


def _map_status_error(error: openai.APIStatusError) -> ProviderError:
    status_code = error.status_code
    if status_code == 401:
        return _status_error(AgentFailureCode.PROVIDER_AUTHENTICATION, error)
    if status_code == 403:
        return _status_error(AgentFailureCode.PROVIDER_PERMISSION, error)
    if status_code == 429:
        return _status_error(AgentFailureCode.PROVIDER_RATE_LIMITED, error)
    if status_code >= 500:
        return _status_error(AgentFailureCode.PROVIDER_SERVER, error)
    return _status_error(AgentFailureCode.PROVIDER_INVALID_REQUEST, error)


def _status_error(code: AgentFailureCode, error: openai.APIStatusError) -> ProviderError:
    headers: Mapping[str, str] | None = getattr(error.response, "headers", None)
    return ProviderError(
        code,
        status_code=error.status_code,
        retry_after_seconds=_retry_after(headers),
    )


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            return None
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return seconds if seconds >= 0 else None
