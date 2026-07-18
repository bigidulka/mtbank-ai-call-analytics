"""Закрытый registry типизированных agent tools без dynamic execution."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from pydantic import BaseModel, ValidationError

from mtbank_ai.agent_runtime.contracts import (
    AgentFailureCode,
    AgentRuntimeError,
    FunctionToolSchema,
    ModelToolCall,
    ToolExecutionContext,
    ToolObservation,
    ToolSideEffect,
)

ToolResult: TypeAlias = BaseModel | Mapping[str, object]
ToolHandler = Callable[[Any, ToolExecutionContext], Awaitable[ToolResult]]


class ToolValidationError(AgentRuntimeError):
    """Модель предложила невалидный tool call."""


class ToolExecutionError(AgentRuntimeError):
    """Доверенный handler не вернул допустимое наблюдение."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Единый источник input/output schema и разрешённого handler."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    side_effect: ToolSideEffect
    timeout_seconds: float
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 256:
            raise ValueError("tool name должен быть непустым и ограниченным")
        if not self.description or len(self.description) > 20_000:
            raise ValueError("tool description должен быть непустым и ограниченным")
        if self.timeout_seconds <= 0:
            raise ValueError("tool timeout должен быть положительным")
        if not isinstance(self.side_effect, ToolSideEffect):
            raise TypeError("tool side_effect должен быть разрешённым ToolSideEffect")
        if not callable(self.handler):
            raise TypeError("tool handler должен быть callable")
        if not _is_async_handler(self.handler):
            raise TypeError("tool handler должен быть async function")
        if not isinstance(self.input_model, type) or not isinstance(self.output_model, type):
            raise TypeError("tool input и output должны быть Pydantic-моделями")
        if not issubclass(self.input_model, BaseModel) or not issubclass(self.output_model, BaseModel):
            raise TypeError("tool input и output должны быть Pydantic-моделями")

    def function_schema(self) -> FunctionToolSchema:
        schema = self.input_model.model_json_schema(mode="validation")
        _make_schema_strict(schema)
        return FunctionToolSchema(name=self.name, description=self.description, parameters=schema)


@dataclass(frozen=True, slots=True)
class ValidatedToolCall:
    call: ModelToolCall
    spec: ToolSpec
    arguments: BaseModel


@dataclass(frozen=True, slots=True)
class ExecutedToolCall:
    output: BaseModel
    observation: ToolObservation | None


class ToolRegistry:
    """Статический allowlist tools; строки модели никогда не становятся кодом."""

    def __init__(self, specs: Sequence[ToolSpec]) -> None:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("имена tools должны быть уникальны")
        self._by_name = by_name

    def require(self, name: str) -> ToolSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise ToolValidationError(AgentFailureCode.UNKNOWN_TOOL) from None

    def function_schemas(self, names: Sequence[str]) -> tuple[FunctionToolSchema, ...]:
        if len(set(names)) != len(names):
            raise ValueError("имена tools должны быть уникальны")
        return tuple(self.require(name).function_schema() for name in names)

    def validate_calls(
        self,
        calls: Sequence[ModelToolCall],
        *,
        seen_remote_call_ids: set[str] | None = None,
    ) -> tuple[ValidatedToolCall, ...]:
        if not calls:
            raise ToolValidationError(AgentFailureCode.MALFORMED_PROVIDER_RESPONSE)
        call_ids = tuple(call.id for call in calls)
        if len(set(call_ids)) != len(call_ids) or (
            seen_remote_call_ids is not None and not set(call_ids).isdisjoint(seen_remote_call_ids)
        ):
            raise ToolValidationError(AgentFailureCode.DUPLICATE_TOOL_CALL_ID)

        validated: list[ValidatedToolCall] = []
        for call in calls:
            spec = self.require(call.name)
            try:
                decoded = json.loads(call.arguments_json)
            except (TypeError, json.JSONDecodeError):
                raise ToolValidationError(AgentFailureCode.TOOL_ARGUMENTS_INVALID) from None
            if not isinstance(decoded, dict):
                raise ToolValidationError(AgentFailureCode.TOOL_ARGUMENTS_INVALID)
            try:
                # JSON-представления UUID, enum и tuple допустимы для строгих
                # domain-моделей; strict validation Python-объекта отклонила бы
                # значения, которые tool-capable provider передаёт только как JSON.
                arguments = spec.input_model.model_validate_json(call.arguments_json, strict=True)
            except ValidationError:
                raise ToolValidationError(AgentFailureCode.TOOL_ARGUMENTS_INVALID) from None
            validated.append(ValidatedToolCall(call=call, spec=spec, arguments=arguments))
        if seen_remote_call_ids is not None:
            seen_remote_call_ids.update(call_ids)
        return tuple(validated)

    async def execute(
        self,
        call: ValidatedToolCall,
        context: ToolExecutionContext,
        *,
        timeout_seconds: float,
        max_observation_bytes: int,
        include_observation: bool = True,
    ) -> ExecutedToolCall:
        try:
            async with asyncio.timeout(timeout_seconds):
                candidate = await call.spec.handler(call.arguments, context)
        except TimeoutError:
            raise ToolExecutionError(AgentFailureCode.TOOL_TIMEOUT) from None
        except AgentRuntimeError:
            raise
        except Exception:
            raise ToolExecutionError(AgentFailureCode.TOOL_EXECUTION_FAILED) from None

        try:
            if isinstance(candidate, BaseModel):
                output = call.spec.output_model.model_validate(candidate.model_dump(), strict=True)
            elif isinstance(candidate, Mapping):
                output = call.spec.output_model.model_validate(candidate, strict=True)
            else:
                raise TypeError("handler должен вернуть Pydantic-модель или mapping")
        except (TypeError, ValidationError):
            raise ToolExecutionError(AgentFailureCode.TOOL_EXECUTION_FAILED) from None

        observation = None
        if include_observation:
            observation = serialize_untrusted_observation(
                tool_name=call.spec.name,
                output=output,
                max_bytes=max_observation_bytes,
            )
        return ExecutedToolCall(output=output, observation=observation)


def _is_async_handler(handler: object) -> bool:
    return inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "__call__", None))


def serialize_untrusted_observation(
    *,
    tool_name: str,
    output: BaseModel,
    max_bytes: int,
) -> ToolObservation:
    """Сериализует tool result для следующего model turn, не для event storage."""

    if max_bytes <= 0:
        raise ValueError("max_bytes должен быть положительным")
    document = {"untrusted_tool_result": output.model_dump(mode="json")}
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ToolExecutionError(AgentFailureCode.TOOL_EXECUTION_FAILED) from None
    if len(encoded) > max_bytes:
        raise ToolExecutionError(AgentFailureCode.OBSERVATION_TOO_LARGE)
    content = encoded.decode("utf-8")
    return ToolObservation(
        tool_name=tool_name,
        observation_hash=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
        untrusted_content=content,
    )


def _make_schema_strict(value: object) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["additionalProperties"] = False
            value["required"] = sorted(properties)
        for child in value.values():
            _make_schema_strict(child)
    elif isinstance(value, list):
        for child in value:
            _make_schema_strict(child)
