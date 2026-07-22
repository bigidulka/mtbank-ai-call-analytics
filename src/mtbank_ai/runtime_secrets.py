"""Проверяет runtime-секреты до запуска сервисов."""

from __future__ import annotations

import re
from collections.abc import Mapping

RUNTIME_SECRET_NAMES = (
    "WEBUI_ADMIN_PASSWORD",
    "WEBUI_SECRET_KEY",
    "PIPELINES_API_KEY",
    "MTBANK_ATTACHMENT_SIGNING_KEY",
    "MTBANK_API_KEY",
    "POSTGRES_PASSWORD",
)
MIN_RUNTIME_SECRET_LENGTH = 32
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

_UNSAFE_EXACT_VALUES = frozenset(
    {
        "0p3n-w3bu!",
        "changeme",
        "default",
        "example",
        "password",
        "secret",
        "test",
    }
)
_PLACEHOLDER_MARKERS = (
    "change",
    "default",
    "example",
    "password",
    "placeholder",
    "replace",
    "secret",
    "token",
    "your",
)


class SecretConfigurationError(ValueError):
    """Runtime-секрет отсутствует или не соответствует безопасному формату."""


def require_runtime_secret(name: str, value: str | None) -> str:
    """Возвращает безопасный секрет либо сообщает только имя некорректной настройки."""

    if not isinstance(value, str) or not _is_safe_secret(value):
        raise SecretConfigurationError(f"{name} отсутствует, слишком короткий или небезопасный")
    return value


def require_environment_secret(name: str, environment: Mapping[str, str | None]) -> str:
    """Loads a safe secret from a deliberately named environment variable."""

    if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
        raise SecretConfigurationError("имя переменной окружения секрета небезопасно")
    return require_runtime_secret(name, environment.get(name))


def validate_runtime_secrets(environment: Mapping[str, str | None]) -> None:
    """Проверяет все обязательные секреты, не раскрывая их значения."""

    invalid_names = [
        name
        for name in RUNTIME_SECRET_NAMES
        if not _is_safe_secret(environment.get(name))
    ]
    if invalid_names:
        raise SecretConfigurationError(
            "Небезопасные runtime-секреты: " + ", ".join(invalid_names)
        )


def _is_safe_secret(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) < MIN_RUNTIME_SECRET_LENGTH:
        return False
    if value != value.strip() or any(character.isspace() for character in value):
        return False

    normalised = value.casefold()
    if normalised in _UNSAFE_EXACT_VALUES:
        return False
    if any(marker in normalised for marker in _PLACEHOLDER_MARKERS):
        return False
    return not _is_repeated_sequence(value)


def _is_repeated_sequence(value: str) -> bool:
    return len(value) > 1 and value in (value + value)[1:-1]
