"""Стабильная таксономия публичных ошибок без раскрытия причин."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from uuid import UUID

from mtbank_ai.domain.base import FrozenModel, LongText


class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    UNSUPPORTED_MEDIA = "unsupported_media"
    INVALID_REQUEST = "invalid_request"
    INVALID_URL = "invalid_url"
    INVALID_AUDIO = "invalid_audio"
    NO_SPEECH = "no_speech"
    ROLE_RESOLUTION_REQUIRED = "role_resolution_required"
    QUOTA_EXCEEDED = "quota_exceeded"
    PROVIDER_FAILURE = "provider_failure"
    AGENT_FAILURE = "agent_failure"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ErrorSpec:
    status_code: int
    message: str
    retryable: bool


ERROR_SPECS = MappingProxyType(
    {
        ErrorCode.INVALID_INPUT: ErrorSpec(400, "Нужно передать ровно один источник аудио.", False),
        ErrorCode.UNAUTHENTICATED: ErrorSpec(401, "Требуется аутентификация.", False),
        ErrorCode.FORBIDDEN: ErrorSpec(403, "Доступ запрещён.", False),
        ErrorCode.PAYLOAD_TOO_LARGE: ErrorSpec(413, "Аудиофайл превышает допустимый размер.", False),
        ErrorCode.UNSUPPORTED_MEDIA: ErrorSpec(415, "Тип содержимого не поддерживается.", False),
        ErrorCode.INVALID_REQUEST: ErrorSpec(422, "Тело запроса некорректно.", False),
        ErrorCode.INVALID_URL: ErrorSpec(422, "URL источника некорректен.", False),
        ErrorCode.INVALID_AUDIO: ErrorSpec(422, "Аудиоданные некорректны.", False),
        ErrorCode.NO_SPEECH: ErrorSpec(422, "В аудиозаписи не обнаружена речь.", False),
        ErrorCode.ROLE_RESOLUTION_REQUIRED: ErrorSpec(409, "Нужно подтвердить роли говорящих.", False),
        ErrorCode.QUOTA_EXCEEDED: ErrorSpec(429, "Квота запросов исчерпана.", True),
        ErrorCode.PROVIDER_FAILURE: ErrorSpec(502, "Внешний провайдер завершил запрос с ошибкой.", True),
        ErrorCode.AGENT_FAILURE: ErrorSpec(502, "Агент анализа завершил запрос с ошибкой.", True),
        ErrorCode.DEADLINE_EXCEEDED: ErrorSpec(504, "Истёк срок выполнения анализа.", True),
        ErrorCode.SERVICE_UNAVAILABLE: ErrorSpec(503, "Сервис анализа пока недоступен.", True),
        ErrorCode.INTERNAL_ERROR: ErrorSpec(500, "Внутренняя ошибка сервиса.", False),
    }
)


class DomainError(Exception):
    """Контролируемая ошибка с фиксированным публичным представлением."""

    def __init__(self, code: ErrorCode) -> None:
        self.code = code
        spec = ERROR_SPECS[code]
        self.status_code = spec.status_code
        self.retryable = spec.retryable
        super().__init__(code.value)


class ApiError(FrozenModel):
    code: ErrorCode
    message: LongText
    request_id: UUID
    retryable: bool


class ErrorResponse(FrozenModel):
    error: ApiError


def build_error_response(error: DomainError, request_id: UUID) -> tuple[int, ErrorResponse]:
    spec = ERROR_SPECS[error.code]
    return (
        spec.status_code,
        ErrorResponse(
            error=ApiError(
                code=error.code,
                message=spec.message,
                request_id=request_id,
                retryable=spec.retryable,
            )
        ),
    )
