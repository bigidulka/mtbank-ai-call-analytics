"""Локальный reviewed prompt registry с canonical hashes и path containment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

from mtbank_ai.agent_runtime.contracts import FunctionToolSchema, PromptReference
from mtbank_ai.domain.base import Sha256, StrictFrozenModel

_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class PromptRegistryError(ValueError):
    """Prompt не является reviewed regular file внутри registry root."""


class PromptBundle(StrictFrozenModel):
    reference: PromptReference
    text: Annotated[str, Field(min_length=1, max_length=20_000)]
    policy_hash: Sha256
    tool_schema_hash: Sha256


class PromptRegistry:
    """Читает только versioned prompt files, не пути от model или request payload."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise PromptRegistryError("prompt registry root не может быть symlink")
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            raise PromptRegistryError("prompt registry root недоступен") from None
        if not resolved_root.is_dir():
            raise PromptRegistryError("prompt registry root должен быть реальным каталогом")
        self._root = resolved_root

    def load(
        self,
        prompt_id: str,
        version: str,
        *,
        policy_inputs: Mapping[str, object] | BaseModel,
        tool_schemas: Sequence[FunctionToolSchema],
    ) -> PromptBundle:
        _validate_component(prompt_id)
        _validate_component(version)
        path = self._prompt_path(prompt_id, version)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise PromptRegistryError("reviewed prompt недоступен") from None
        canonical_text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not canonical_text.strip():
            raise PromptRegistryError("reviewed prompt не может быть пустым")

        content_hash = _hash_text(canonical_text)
        policy_hash = _hash_json(_model_or_mapping(policy_inputs))
        tool_schema_hash = _hash_json(tuple(tool.model_dump(mode="json") for tool in tool_schemas))
        bundle_hash = _hash_json(
            {
                "content_hash": content_hash,
                "policy_hash": policy_hash,
                "prompt_id": prompt_id,
                "tool_schema_hash": tool_schema_hash,
                "version": version,
            }
        )
        try:
            return PromptBundle(
                reference=PromptReference(
                    prompt_id=prompt_id,
                    version=version,
                    content_hash=content_hash,
                    bundle_hash=bundle_hash,
                ),
                text=canonical_text,
                policy_hash=policy_hash,
                tool_schema_hash=tool_schema_hash,
            )
        except ValidationError:
            raise PromptRegistryError("reviewed prompt превышает допустимый размер") from None

    def _prompt_path(self, prompt_id: str, version: str) -> Path:
        candidate = self._root / prompt_id / f"{version}.md"
        relative = candidate.relative_to(self._root)
        current = self._root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise PromptRegistryError("symlink в prompt path запрещён")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise PromptRegistryError("reviewed prompt отсутствует") from None
        if not resolved.is_file() or not resolved.is_relative_to(self._root):
            raise PromptRegistryError("prompt path выходит за registry root")
        return resolved


def _validate_component(value: str) -> None:
    if not _COMPONENT.fullmatch(value):
        raise PromptRegistryError("prompt ID и version должны быть простыми безопасными компонентами")


def _model_or_mapping(value: Mapping[str, object] | BaseModel) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise PromptRegistryError("policy и schema inputs должны быть canonical JSON") from None
    return hashlib.sha256(encoded).hexdigest()
