"""Bounded typed LLM role resolver with reviewed prompt provenance."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx
import openai
from openai import OpenAI
from pydantic import ValidationError

from mtbank_ai.agent_runtime import MessageRole, ModelMessage, ModelRequest, PromptRegistry, ToolChoice
from mtbank_ai.agent_runtime.contracts import FunctionToolSchema, ModelResponse, ModelToolCall, ModelUsage
from mtbank_ai.domain.transcript import RoleAgentProvenance
from mtbank_ai.speech.contracts import RoleResolutionCandidate, RoleResolutionDecision
from services.speech.errors import SpeechProviderError
from services.speech.settings import RoleAgentSettings

_PROMPT_ID = "role_resolver"
_PROMPT_VERSION = "v1"
_PROMPT_ROOT = Path(__file__).resolve().parents[2] / "src" / "mtbank_ai" / "agents"
_TOOL_NAME = "submit_role_resolution"


class RoleModelPort(Protocol):
    def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse: ...

    def close(self) -> None: ...


class SyncRoleModelProvider:
    """Single synchronous Chat Completions boundary for speech worker threads."""

    def __init__(self, settings: RoleAgentSettings, *, client: Any | None = None) -> None:
        self._client = client or OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=str(settings.base_url),
            timeout=httpx.Timeout(settings.timeout_seconds, connect=settings.connect_timeout_seconds),
            max_retries=0,
        )

    def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        remaining = (deadline_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise SpeechProviderError("role agent deadline exceeded")
        try:
            completion = self._client.chat.completions.create(
                model=request.model_id,
                messages=[_message_payload(message) for message in request.messages],  # pyright: ignore[reportArgumentType]
                tools=[_tool_payload(tool) for tool in request.tools],  # pyright: ignore[reportArgumentType]
                tool_choice=request.tool_choice.value,
                max_tokens=request.max_output_tokens,
                temperature=request.temperature,
                timeout=remaining,
            )
            choice = completion.choices[0]
            message = choice.message
            usage = completion.usage
            if usage is None or not completion.model:
                raise ValueError("role agent provider omitted usage or model")
            calls = tuple(_model_tool_call(call) for call in (message.tool_calls or ()))
            text = message.content.strip() if isinstance(message.content, str) and message.content.strip() else None
            return ModelResponse(
                request_id=None,
                model_id=completion.model,
                finish_reason=choice.finish_reason,
                tool_calls=calls,
                usage=ModelUsage(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                ),
                latency_ms=0,
                has_text_content=text is not None,
                text_content=text,
            )
        except SpeechProviderError:
            raise
        except (openai.APIError, AttributeError, IndexError, TypeError, ValueError) as error:
            raise SpeechProviderError("role agent provider request failed") from error

    def close(self) -> None:
        self._client.close()


class LlmRoleResolver:
    """One model call, one typed terminal tool, no heuristic or alternate provider."""

    def __init__(
        self,
        settings: RoleAgentSettings,
        *,
        provider: RoleModelPort | None = None,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider or SyncRoleModelProvider(settings)
        self._prompt_registry = prompt_registry or PromptRegistry(_PROMPT_ROOT)

    def resolve(self, candidates: tuple[RoleResolutionCandidate, ...]) -> RoleResolutionDecision:
        if not candidates:
            return RoleResolutionDecision()
        tool = FunctionToolSchema(
            name=_TOOL_NAME,
            description="Submit complete role assignments or an empty roles list when ambiguous.",
            parameters=RoleResolutionDecision.model_json_schema(),
        )
        prompt = self._prompt_registry.load(
            _PROMPT_ID,
            _PROMPT_VERSION,
            policy_inputs={
                "max_candidates": self._settings.max_candidates,
                "max_input_chars": self._settings.max_input_chars,
                "max_output_tokens": self._settings.max_output_tokens,
                "tools": [_TOOL_NAME],
            },
            tool_schemas=(tool,),
        )
        payload = json.dumps(
            {"candidates": [candidate.model_dump(mode="json") for candidate in candidates]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(candidates) > self._settings.max_candidates or len(payload) > self._settings.max_input_chars:
            return RoleResolutionDecision()
        response = self._provider.complete(
            ModelRequest(
                model_id=self._settings.model,
                messages=(
                    ModelMessage(role=MessageRole.SYSTEM, content=prompt.text),
                    ModelMessage(role=MessageRole.USER, content=payload),
                ),
                tools=(tool,),
                tool_choice=ToolChoice.REQUIRED,
                max_output_tokens=self._settings.max_output_tokens,
                temperature=0.0,
            ),
            deadline_at=datetime.now(UTC) + timedelta(seconds=self._settings.timeout_seconds),
        )
        if response.model_id != self._settings.model or response.has_text_content or len(response.tool_calls) != 1:
            raise SpeechProviderError("role agent returned invalid terminal response")
        call = response.tool_calls[0]
        if call.name != _TOOL_NAME:
            raise SpeechProviderError("role agent called an unauthorized tool")
        try:
            submitted = RoleResolutionDecision.model_validate_json(call.arguments_json, strict=True)
        except ValidationError as error:
            repaired = self._retry_invalid_output(
                prompt_text=prompt.text,
                payload=payload,
                tool=tool,
                invalid_arguments=call.arguments_json,
            )
            if repaired is None:
                raise SpeechProviderError("role agent returned invalid typed output") from error
            submitted = repaired
        return submitted.model_copy(
            update={
                "agent_provenance": RoleAgentProvenance(
                    policy_id=_PROMPT_ID,
                    version=_PROMPT_VERSION,
                    owner="MTBank AI Engineering",
                    effective_date=self._settings.prompt_effective_date,
                    sha256=prompt.reference.bundle_hash,
                )
            }
        )

    def _retry_invalid_output(
        self,
        *,
        prompt_text: str,
        payload: str,
        tool: FunctionToolSchema,
        invalid_arguments: str,
    ) -> RoleResolutionDecision | None:
        response = self._provider.complete(
            ModelRequest(
                model_id=self._settings.model,
                messages=(
                    ModelMessage(role=MessageRole.SYSTEM, content=prompt_text),
                    ModelMessage(role=MessageRole.USER, content=payload),
                    ModelMessage(
                        role=MessageRole.USER,
                        content=(
                            "Предыдущий JSON не прошёл схему. Исправь его и вызови "
                            f"{_TOOL_NAME} ещё раз. Не добавляй текст. Невалидный JSON: "
                            f"{invalid_arguments[: self._settings.max_input_chars // 2]}"
                        ),
                    ),
                ),
                tools=(tool,),
                tool_choice=ToolChoice.REQUIRED,
                max_output_tokens=self._settings.max_output_tokens,
                temperature=0.0,
            ),
            deadline_at=datetime.now(UTC) + timedelta(seconds=self._settings.timeout_seconds),
        )
        if response.model_id != self._settings.model or response.has_text_content or len(response.tool_calls) != 1:
            return None
        call = response.tool_calls[0]
        if call.name != _TOOL_NAME:
            return None
        try:
            return RoleResolutionDecision.model_validate_json(call.arguments_json, strict=True)
        except ValidationError:
            return None

    def close(self) -> None:
        self._provider.close()


def _model_tool_call(call: object) -> ModelToolCall:
    function = getattr(call, "function", None)
    return ModelToolCall(
        id=str(getattr(call, "id", "")),
        name=str(getattr(function, "name", "")),
        arguments_json=str(getattr(function, "arguments", "")),
    )


def _message_payload(message: ModelMessage) -> dict[str, object]:
    return {"role": message.role.value, "content": message.content}


def _tool_payload(tool: FunctionToolSchema) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
