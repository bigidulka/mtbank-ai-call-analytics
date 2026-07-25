"""Bounded text-only demo assistant over the configured OpenAI-compatible gateway."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator

from mtbank_ai.agent_runtime import (
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PromptRegistry,
    ToolChoice,
)
from mtbank_ai.config import AgentRuntimeSettings
from mtbank_ai.domain.base import StrictFrozenModel

_PROMPT_ID = "demo_assistant"
_PROMPT_VERSION = "v1"
_PROMPT_ROOT = Path(__file__).resolve().parent / "agents"
_MAX_HISTORY_MESSAGES = 8
_MAX_MESSAGE_CHARS = 2_000
_MAX_OUTPUT_TOKENS = 500
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


class AssistantModelPort(Protocol):
    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse: ...

    async def close(self) -> None: ...


class DemoAssistant:
    def __init__(
        self,
        model_client: AssistantModelPort,
        runtime_settings: AgentRuntimeSettings,
        *,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._model_client = model_client
        self._runtime_settings = runtime_settings
        self._prompt_registry = prompt_registry or PromptRegistry(_PROMPT_ROOT)

    async def answer(self, request: AssistantRequest) -> AssistantResponse:
        prompt = self._prompt_registry.load(
            _PROMPT_ID,
            _PROMPT_VERSION,
            policy_inputs={
                "deadline_seconds": min(_DEADLINE_SECONDS, self._runtime_settings.default_deadline_seconds),
                "history_messages": _MAX_HISTORY_MESSAGES,
                "message_chars": _MAX_MESSAGE_CHARS,
                "output_tokens": min(_MAX_OUTPUT_TOKENS, self._runtime_settings.default_max_output_tokens),
                "tools_enabled": False,
            },
            tool_schemas=(),
        )
        messages = [ModelMessage(role=MessageRole.SYSTEM, content=prompt.text)]
        messages.extend(ModelMessage(role=MessageRole(item.role), content=item.content) for item in request.history)
        messages.append(ModelMessage(role=MessageRole.USER, content=request.message))
        model_id = self._runtime_settings.gateway.models.default_model
        response = await self._model_client.complete(
            ModelRequest(
                model_id=model_id,
                reasoning_effort=self._runtime_settings.gateway.models.default_reasoning_effort,
                messages=tuple(messages),
                tools=(),
                tool_choice=ToolChoice.NONE,
                max_output_tokens=min(_MAX_OUTPUT_TOKENS, self._runtime_settings.default_max_output_tokens),
                temperature=0.2,
            ),
            deadline_at=datetime.now(UTC)
            + timedelta(seconds=min(_DEADLINE_SECONDS, self._runtime_settings.default_deadline_seconds)),
        )
        if response.model_id != model_id or response.text_content is None:
            raise ValueError("assistant provider вернул некорректный text response")
        return AssistantResponse(answer=response.text_content, model_id=response.model_id)

    async def close(self) -> None:
        await self._model_client.close()
