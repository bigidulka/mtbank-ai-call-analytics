from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import SecretStr, ValidationError

from mtbank_ai.agent_runtime import (
    AgentFailureCode,
    AgentRuntimeError,
    ConfiguredOpenAICompatibleGateway,
    MessageRole,
    ModelMessage,
    ModelRequest,
    OpenAICompatibleProvider,
    ProviderError,
    ToolChoice,
)
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema
from mtbank_ai.agent_runtime.retry import CircuitState
from mtbank_ai.config import GatewayModelSettings, GatewaySettings
from mtbank_ai.domain.base import ReasoningEffort

NOW = datetime(2026, 7, 16, tzinfo=UTC)
SAFE_GATEWAY_KEY = "A7#vM2!qL9@xR4$kT8%hN5^zC1&wP6*eD3"


class FakeClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def close(self) -> None:
        return None


class FakeStream:
    def __init__(self, chunks: tuple[object, ...]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


class StreamingFakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: object) -> FakeStream:
        self.calls.append(kwargs)
        stream = FakeStream(
            (
                SimpleNamespace(
                    model="configured-model",
                    usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
                ),
            )
        )
        self.streams.append(stream)
        return stream

    async def close(self) -> None:
        return None


def _settings(**changes: object) -> GatewaySettings:
    values: dict[str, Any] = {
        "base_url": "https://gateway.example.test/v1",
        "api_key": SecretStr(SAFE_GATEWAY_KEY),
        "models": GatewayModelSettings(default_model="configured-model"),
    }
    values.update(changes)
    return GatewaySettings(**values)


def _request(*, reasoning_effort: ReasoningEffort | None = None) -> ModelRequest:
    return ModelRequest(
        model_id="configured-model",
        reasoning_effort=reasoning_effort,
        messages=(ModelMessage(role=MessageRole.SYSTEM, content="private prompt"),),
        tools=(
            FunctionToolSchema(
                name="lookup",
                description="Lookup evidence.",
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
            ),
        ),
        tool_choice=ToolChoice.REQUIRED,
        max_output_tokens=8,
    )


def _completion(*, request_id: str = "gateway-request-id", tool_call_id: str = "tool-call") -> object:
    return SimpleNamespace(
        choices=(
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=(
                        SimpleNamespace(
                            id=tool_call_id,
                            function=SimpleNamespace(name="lookup", arguments='{"x":"private"}'),
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ),
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        model="configured-model",
        _request_id=request_id,
    )


def test_provider_maps_typed_response_and_uses_chat_completions_only() -> None:
    client = FakeClient(_completion())
    provider = OpenAICompatibleProvider(_settings(), client=client, now=lambda: NOW, monotonic=lambda: 1.0)

    response = asyncio.run(provider.complete(_request(), deadline_at=NOW + timedelta(seconds=10)))

    assert response.request_id == "gateway-request-id"
    assert response.model_id == "configured-model"
    assert response.tool_calls[0].arguments_json == '{"x":"private"}'
    assert response.usage.total_tokens == 5
    assert response.has_text_content is False
    assert client.calls[0]["tool_choice"] == "required"
    assert "strict" not in client.calls[0]["tools"][0]["function"]
    assert "responses" not in vars(client)


@pytest.mark.parametrize("reasoning_effort", ("high", None))
def test_provider_forwards_reasoning_effort_only_when_configured(
    reasoning_effort: ReasoningEffort | None,
) -> None:
    client = FakeClient(_completion())
    provider = OpenAICompatibleProvider(_settings(), client=client, now=lambda: NOW)
    request = _request(reasoning_effort=reasoning_effort)

    asyncio.run(provider.complete(request, deadline_at=NOW + timedelta(seconds=10)))

    streaming_client = StreamingFakeClient()
    streaming_provider = OpenAICompatibleProvider(_settings(), client=streaming_client, now=lambda: NOW)
    streaming_result = asyncio.run(streaming_provider.probe_streaming(request, deadline_at=NOW + timedelta(seconds=10)))

    for payload in (client.calls[0], *streaming_client.calls):
        if reasoning_effort is None:
            assert "reasoning_effort" not in payload
        else:
            assert payload["reasoning_effort"] == reasoning_effort
    assert streaming_result.limit_enforced is True
    assert streaming_client.streams[1].closed is True


def test_model_request_rejects_max_reasoning_effort() -> None:
    payload = _request().model_dump()
    payload["reasoning_effort"] = "max"

    with pytest.raises(ValidationError):
        ModelRequest.model_validate(payload)


def test_provider_disables_sdk_retries_and_does_not_retain_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class ConstructorClient(FakeClient):
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            super().__init__(_completion())

    monkeypatch.setattr("mtbank_ai.agent_runtime.provider.AsyncOpenAI", ConstructorClient)
    provider = OpenAICompatibleProvider(_settings())

    assert captured["max_retries"] == 0
    assert captured["base_url"] == "https://gateway.example.test/v1"
    assert "http_client" not in captured
    assert SAFE_GATEWAY_KEY not in repr(provider)


def test_trusted_local_provider_disables_environment_proxy_for_custom_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_openai: dict[str, object] = {}
    captured_http_client: dict[str, object] = {}
    http_client = object()

    class ConstructorClient(FakeClient):
        def __init__(self, **kwargs: object) -> None:
            captured_openai.update(kwargs)
            super().__init__(_completion())

    def build_http_client(**kwargs: object) -> object:
        captured_http_client.update(kwargs)
        return http_client

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8319")
    monkeypatch.setattr("mtbank_ai.agent_runtime.provider.AsyncOpenAI", ConstructorClient)
    monkeypatch.setattr("mtbank_ai.agent_runtime.provider.httpx.AsyncClient", build_http_client)
    OpenAICompatibleProvider(
        _settings(
            transport_mode="trusted_local_http",
            base_url="http://127.0.0.1:8317/v1",
            request_timeout_seconds=7.0,
            connect_timeout_seconds=2.0,
        )
    )

    timeout = captured_openai["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 2.0
    assert timeout.read == 7.0
    assert timeout.write == 7.0
    assert timeout.pool == 7.0
    assert captured_http_client == {"timeout": timeout, "trust_env": False, "follow_redirects": False}
    assert captured_openai["http_client"] is http_client


def test_trusted_local_provider_rejects_redirect_without_following_it(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []
    actual_async_client = httpx.AsyncClient

    def redirect(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": "http://127.0.0.1:8318/redirect"},
            request=request,
        )

    def build_http_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
        return actual_async_client(
            transport=httpx.MockTransport(redirect),
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        )

    monkeypatch.setattr("mtbank_ai.agent_runtime.provider._trusted_local_http_client", build_http_client)
    provider = OpenAICompatibleProvider(
        _settings(
            transport_mode="trusted_local_http",
            base_url="http://127.0.0.1:8317/v1",
        ),
        now=lambda: NOW,
    )
    try:
        with pytest.raises(ProviderError):
            asyncio.run(provider.complete(_request(), deadline_at=NOW + timedelta(seconds=10)))
    finally:
        asyncio.run(provider.close())

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/chat/completions"


def test_configured_gateway_applies_retry_and_circuit_settings() -> None:
    request = httpx.Request("POST", "https://gateway.example.test/v1/chat/completions")
    response = httpx.Response(500, request=request)
    provider_error = openai.InternalServerError("provider-body", response=response, body={"detail": "provider-body"})
    gateway = ConfiguredOpenAICompatibleGateway(
        _settings(retry_max_attempts=1, circuit_failure_threshold=1),
        client=FakeClient(provider_error),
    )

    with pytest.raises(ProviderError) as error:
        asyncio.run(gateway.complete(_request(), deadline_at=datetime.now(UTC) + timedelta(seconds=10)))

    assert error.value.code is AgentFailureCode.PROVIDER_SERVER
    assert gateway.circuit_breaker.state is CircuitState.OPEN
    asyncio.run(gateway.close())


def test_provider_maps_rate_limit_without_secret_or_response_body() -> None:
    request = httpx.Request("POST", "https://gateway.example.test/v1/chat/completions")
    remote_request_id = "provider-request-secret"
    response = httpx.Response(429, request=request, headers={"retry-after": "2.5", "x-request-id": remote_request_id})
    error = openai.RateLimitError("provider-body-secret", response=response, body={"detail": "provider-body-secret"})
    provider = OpenAICompatibleProvider(_settings(), client=FakeClient(error), now=lambda: NOW)

    with pytest.raises(ProviderError) as mapped:
        asyncio.run(provider.complete(_request(), deadline_at=NOW + timedelta(seconds=10)))

    assert mapped.value.code is AgentFailureCode.PROVIDER_RATE_LIMITED
    assert mapped.value.retry_after_seconds == 2.5
    assert not hasattr(mapped.value, "request_id")
    assert "provider-body-secret" not in str(mapped.value)
    assert remote_request_id not in repr(mapped.value)
    assert SAFE_GATEWAY_KEY not in repr(mapped.value)
    assert mapped.value.__cause__ is None


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            openai.AuthenticationError(
                "private-body",
                response=httpx.Response(401, request=httpx.Request("POST", "https://gateway.example.test")),
                body={"detail": "private-body"},
            ),
            AgentFailureCode.PROVIDER_AUTHENTICATION,
        ),
        (
            openai.PermissionDeniedError(
                "private-body",
                response=httpx.Response(403, request=httpx.Request("POST", "https://gateway.example.test")),
                body={"detail": "private-body"},
            ),
            AgentFailureCode.PROVIDER_PERMISSION,
        ),
        (
            openai.BadRequestError(
                "private-body",
                response=httpx.Response(400, request=httpx.Request("POST", "https://gateway.example.test")),
                body={"detail": "private-body"},
            ),
            AgentFailureCode.PROVIDER_INVALID_REQUEST,
        ),
        (
            openai.BadRequestError(
                "private-body",
                response=httpx.Response(400, request=httpx.Request("POST", "https://gateway.example.test")),
                body={"code": "tool_use_failed", "failed_generation": "private-body"},
            ),
            AgentFailureCode.PROVIDER_TOOL_USE_FAILED,
        ),
        (
            openai.InternalServerError(
                "private-body",
                response=httpx.Response(500, request=httpx.Request("POST", "https://gateway.example.test")),
                body={"detail": "private-body"},
            ),
            AgentFailureCode.PROVIDER_SERVER,
        ),
        (
            openai.APIConnectionError(request=httpx.Request("POST", "https://gateway.example.test")),
            AgentFailureCode.PROVIDER_TRANSPORT,
        ),
        (
            openai.APITimeoutError(request=httpx.Request("POST", "https://gateway.example.test")),
            AgentFailureCode.PROVIDER_TIMEOUT,
        ),
    ),
)
def test_provider_maps_all_typed_error_boundaries(error: BaseException, expected: AgentFailureCode) -> None:
    provider = OpenAICompatibleProvider(_settings(), client=FakeClient(error), now=lambda: NOW)

    with pytest.raises(ProviderError) as mapped:
        asyncio.run(provider.complete(_request(), deadline_at=NOW + timedelta(seconds=10)))

    assert mapped.value.code is expected
    assert "private-body" not in str(mapped.value)


@pytest.mark.parametrize(
    "completion",
    (
        _completion(request_id="r" * 257),
        _completion(tool_call_id="t" * 257),
    ),
)
def test_provider_rejects_unbounded_remote_ids(completion: object) -> None:
    provider = OpenAICompatibleProvider(_settings(), client=FakeClient(completion), now=lambda: NOW)

    with pytest.raises(AgentRuntimeError) as error:
        asyncio.run(provider.complete(_request(), deadline_at=NOW + timedelta(seconds=10)))

    assert error.value.code is AgentFailureCode.MALFORMED_PROVIDER_RESPONSE
    assert "r" * 257 not in str(error.value)
    assert "t" * 257 not in str(error.value)


def test_provider_rejects_malformed_completion_and_invalid_gateway_settings() -> None:
    malformed = SimpleNamespace(choices=(), usage=None, model="configured-model", _request_id="id")
    provider = OpenAICompatibleProvider(_settings(), client=FakeClient(malformed), now=lambda: NOW)
    with pytest.raises(Exception) as error:
        asyncio.run(provider.complete(_request(), deadline_at=NOW + timedelta(seconds=10)))
    assert getattr(error.value, "code") is AgentFailureCode.MALFORMED_PROVIDER_RESPONSE

    with pytest.raises(ValidationError):
        _settings(base_url="http://localhost:8080/v1")
    with pytest.raises(ValidationError):
        _settings(base_url="https://127.0.0.1/v1")
    with pytest.raises(ValidationError):
        _settings(api_key=SecretStr("example-key-value-that-is-long-enough-for-a-test"))
