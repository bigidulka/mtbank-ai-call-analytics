from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr, ValidationError

from mtbank_ai.agent_runtime import ModelRequest, ModelResponse, ModelUsage
from mtbank_ai.assistant import AssistantMessage, AssistantRequest, DemoAssistant
from mtbank_ai.config import AgentRuntimeSettings, GatewayModelSettings, GatewaySettings

SAFE_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


class TextClient:
    def __init__(self, *, text: str = "Осмысленный ответ помощника.", model_id: str = "assistant-model") -> None:
        self.text = text
        self.model_id = model_id
        self.requests: list[ModelRequest] = []
        self.deadlines: list[datetime] = []
        self.closed = False

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        self.requests.append(request)
        self.deadlines.append(deadline_at)
        return ModelResponse(
            request_id="assistant-request",
            model_id=self.model_id,
            finish_reason="stop",
            tool_calls=(),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            latency_ms=12,
            has_text_content=True,
            text_content=self.text,
        )

    async def close(self) -> None:
        self.closed = True


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


def test_demo_assistant_uses_reviewed_prompt_and_real_model_with_bounded_history_and_no_tools() -> None:
    client = TextClient()
    assistant = DemoAssistant(client, _runtime())
    request = AssistantRequest(
        message="Что ты умеешь?",
        history=tuple(
            AssistantMessage(role="user" if index % 2 == 0 else "assistant", content=f"message-{index}")
            for index in range(8)
        ),
    )

    response = asyncio.run(assistant.answer(request))

    model_request = client.requests[0]
    assert response.answer == "Осмысленный ответ помощника."
    assert response.model_id == "assistant-model"
    assert model_request.model_id == "assistant-model"
    assert model_request.tool_choice.value == "none"
    assert model_request.tools == ()
    assert model_request.max_output_tokens == 500
    assert model_request.temperature == 0.2
    assert len(model_request.messages) == 10
    assert model_request.messages[0].role.value == "system"
    assert model_request.messages[-1].content == "Что ты умеешь?"
    assert model_request.messages[0].content is not None
    assert "secrets" in model_request.messages[0].content
    assert "text-only LLM-agent" in model_request.messages[0].content
    assert 0 < (client.deadlines[0] - datetime.now(UTC)).total_seconds() <= 15


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
    wrong_model = DemoAssistant(TextClient(model_id="unconfigured-model"), _runtime())
    with pytest.raises(ValueError, match="text response"):
        asyncio.run(wrong_model.answer(AssistantRequest(message="hi")))

    class EmptyClient(TextClient):
        async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
            del request, deadline_at
            return ModelResponse(
                request_id=None,
                model_id="assistant-model",
                finish_reason="stop",
                tool_calls=(),
                usage=ModelUsage(input_tokens=1, output_tokens=0, total_tokens=1),
                latency_ms=1,
                has_text_content=False,
            )

    with pytest.raises(ValueError, match="text response"):
        asyncio.run(DemoAssistant(EmptyClient(), _runtime()).answer(AssistantRequest(message="hi")))
