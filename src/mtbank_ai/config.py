"""Typed runtime-настройки нового application foundation."""

from __future__ import annotations

from decimal import Decimal
from ipaddress import ip_address
from typing import Any, Literal, Self, TypeAlias
from urllib.parse import unquote, urlsplit

from pydantic import HttpUrl, SecretStr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mtbank_ai.domain.base import (
    MimeType,
    NonEmptyId,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ReasoningEffort,
    StrictFrozenModel,
)
from mtbank_ai.runtime_secrets import SecretConfigurationError, require_runtime_secret


class ApiSettings(StrictFrozenModel):
    host: NonEmptyId = "0.0.0.0"
    port: PositiveInt = 8000
    api_key: SecretStr
    max_json_bytes: PositiveInt = 1 * 1024 * 1024
    max_upload_bytes: PositiveInt = 25 * 1024 * 1024
    multipart_reserve_bytes: NonNegativeInt = 64 * 1024
    allowed_media_types: tuple[MimeType, ...] = (
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/ogg",
    )
    allowed_url_schemes: tuple[NonEmptyId, ...] = ("http", "https")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        api_key = value.get_secret_value()
        if not api_key.isascii():
            raise ValueError("API key должен содержать только ASCII символы")
        try:
            require_runtime_secret("MTBANK_API__API_KEY", api_key)
        except SecretConfigurationError as error:
            raise ValueError("API key отсутствует, слишком короткий или небезопасный") from error
        return value

    @field_validator("allowed_media_types", "allowed_url_schemes", mode="before")
    @classmethod
    def parse_environment_sequence(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("allowed_url_schemes")
    @classmethod
    def validate_allowed_url_schemes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or len(set(values)) != len(values) or not set(values).issubset({"http", "https"}):
            raise ValueError("разрешены только уникальные URL schemes http и https")
        return values


def _is_non_cloud_ip(host: str) -> bool:
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local or address.is_unspecified


def _is_non_public_direct_ip(host: str) -> bool:
    try:
        return not ip_address(host).is_global
    except ValueError:
        return False


def _is_internal_direct_ip(host: str) -> bool:
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _is_internal_service_name(host: str) -> bool:
    normalized = host.casefold().removesuffix(".")
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(".localhost"):
        return True
    labels = normalized.split(".")
    if len(labels) == 1:
        return _is_dns_label(labels[0])
    return labels[-1] == "internal" and all(_is_dns_label(label) for label in labels)


def _is_dns_label(value: str) -> bool:
    return (
        1 <= len(value) <= 63
        and value.isascii()
        and value[0].isalnum()
        and value[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in value)
    )


def _is_non_public_speech_dns_name(host: str) -> bool:
    normalized = host.casefold().removesuffix(".")
    return _is_internal_service_name(normalized) or normalized == "local" or normalized.endswith(".local")


SpeechTransportMode: TypeAlias = Literal["internal_http", "remote_https"]
GatewayTransportMode: TypeAlias = Literal["cloud_https", "trusted_local_http"]
_SPEECH_MAX_SUCCESS_RESPONSE_BYTES = 4 * 1024 * 1024
_SPEECH_MAX_ERROR_RESPONSE_BYTES = 64 * 1024


def _validate_speech_response_limits(max_success_response_bytes: int, max_error_response_bytes: int) -> None:
    if max_success_response_bytes > _SPEECH_MAX_SUCCESS_RESPONSE_BYTES:
        raise ValueError("max_success_response_bytes не может превышать 4194304")
    if max_error_response_bytes > _SPEECH_MAX_ERROR_RESPONSE_BYTES:
        raise ValueError("max_error_response_bytes не может превышать 65536")
    if max_error_response_bytes >= max_success_response_bytes:
        raise ValueError("max_error_response_bytes должен быть меньше max_success_response_bytes")


def _validate_speech_base_url(value: HttpUrl, mode: SpeechTransportMode) -> HttpUrl:
    try:
        parts = urlsplit(str(value))
        parts.port
    except ValueError:
        raise ValueError("speech base_url имеет некорректную authority") from None
    if (
        not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("speech base_url не может содержать credentials, query или fragment")
    if mode == "internal_http":
        if parts.scheme != "http" or not (
            _is_internal_direct_ip(parts.hostname) or _is_internal_service_name(parts.hostname)
        ):
            raise ValueError("internal speech должен использовать HTTP URL internal host")
        return value
    if (
        parts.scheme != "https"
        or _is_non_public_speech_dns_name(parts.hostname)
        or _is_non_public_direct_ip(parts.hostname)
    ):
        raise ValueError("remote speech должен использовать public HTTPS URL")
    return value


def _require_speech_api_key(mode: SpeechTransportMode, value: SecretStr | None) -> None:
    if mode == "internal_http":
        if value is not None:
            raise ValueError("internal speech API key запрещён")
        return
    if value is None:
        raise ValueError("remote speech API key обязателен")
    api_key = value.get_secret_value()
    if not api_key.isascii():
        raise ValueError("speech API key должен содержать только ASCII символы")
    try:
        require_runtime_secret("MTBANK_SPEECH__API_KEY", api_key)
    except SecretConfigurationError as error:
        raise ValueError("speech API key отсутствует, слишком короткий или небезопасный") from error


def _validate_speech_service_path(value: str) -> str:
    if value != value.strip() or "?" in value or "#" in value:
        raise ValueError("speech service path не может содержать query или fragment")
    parts = urlsplit(value)
    if parts.scheme or parts.netloc or not parts.path.startswith("/"):
        raise ValueError("speech service path должен быть абсолютным путём")
    decoded_path = parts.path
    for _ in range(8):
        next_path = unquote(decoded_path)
        if next_path == decoded_path:
            break
        decoded_path = next_path
    if (
        "%" in decoded_path
        or "\\" in decoded_path
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in decoded_path)
        or any(part in {".", ".."} for part in decoded_path.split("/"))
    ):
        raise ValueError("speech service path не может содержать dot traversal")
    return parts.path


def _validate_speech_transcription_path(value: str) -> str:
    """Совместимый alias для batch speech client."""

    return _validate_speech_service_path(value)


def _is_trusted_local_host(host: str) -> bool:
    return host.casefold() in {"127.0.0.1", "::1", "localhost"}


_LOCAL_GATEWAY_CREDENTIAL_MIN_LENGTH = 8
_LOCAL_GATEWAY_CREDENTIAL_MAX_LENGTH = 256


def _is_safe_local_gateway_credential(value: str) -> bool:
    return (
        _LOCAL_GATEWAY_CREDENTIAL_MIN_LENGTH <= len(value) <= _LOCAL_GATEWAY_CREDENTIAL_MAX_LENGTH
        and value.isascii()
        and not any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
    )


class GatewayModelSettings(StrictFrozenModel):
    default_model: NonEmptyId
    default_reasoning_effort: ReasoningEffort | None = None
    classifier_model: NonEmptyId | None = None
    classifier_reasoning_effort: ReasoningEffort | None = None
    quality_model: NonEmptyId | None = None
    quality_reasoning_effort: ReasoningEffort | None = None
    compliance_model: NonEmptyId | None = None
    compliance_reasoning_effort: ReasoningEffort | None = None
    summarizer_model: NonEmptyId | None = None
    summarizer_reasoning_effort: ReasoningEffort | None = None
    trends_model: NonEmptyId | None = None
    trends_reasoning_effort: ReasoningEffort | None = None
    capability_probe_model: NonEmptyId | None = None
    input_token_cost_usd: NonNegativeDecimal = Decimal("0")
    output_token_cost_usd: NonNegativeDecimal = Decimal("0")


class GatewaySettings(StrictFrozenModel):
    """Конфигурация единственного OpenAI-compatible gateway."""

    transport_mode: GatewayTransportMode = "cloud_https"
    base_url: str
    api_key: SecretStr
    models: GatewayModelSettings
    request_timeout_seconds: PositiveFloat = 20.0
    connect_timeout_seconds: PositiveFloat = 5.0
    max_concurrency: PositiveInt = 4
    retry_max_attempts: PositiveInt = 3
    retry_base_delay_seconds: PositiveFloat = 0.25
    retry_max_delay_seconds: PositiveFloat = 2.0
    retry_max_retry_after_seconds: PositiveFloat = 5.0
    circuit_failure_threshold: PositiveInt = 3
    circuit_recovery_seconds: PositiveFloat = 10.0

    @field_validator("base_url")
    @classmethod
    def validate_gateway_url(cls, value: str, info: ValidationInfo) -> str:
        try:
            parts = urlsplit(value)
            parts.port
        except ValueError:
            raise ValueError("gateway base_url имеет некорректную authority") from None
        if (
            value != value.strip()
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("gateway base_url не может содержать credentials, query или fragment")
        if info.data.get("transport_mode") == "trusted_local_http":
            if parts.scheme != "http" or not _is_trusted_local_host(parts.hostname):
                raise ValueError("trusted local gateway должен быть HTTP URL loopback host")
        elif (
            parts.scheme != "https"
            or parts.hostname.casefold() in {"localhost", "localhost.localdomain"}
            or _is_non_cloud_ip(parts.hostname)
        ):
            raise ValueError("gateway base_url должен быть HTTPS URL облачного gateway")
        return value.rstrip("/")

    @field_validator("api_key")
    @classmethod
    def validate_gateway_api_key(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        api_key = value.get_secret_value()
        if info.data.get("transport_mode") == "trusted_local_http":
            if not _is_safe_local_gateway_credential(api_key):
                raise ValueError("trusted local gateway API key имеет небезопасный формат")
            return value
        if not api_key.isascii():
            raise ValueError("gateway API key должен содержать только ASCII символы")
        try:
            require_runtime_secret("MTBANK_AGENT_RUNTIME__GATEWAY__API_KEY", api_key)
        except SecretConfigurationError as error:
            raise ValueError("gateway API key отсутствует, слишком короткий или небезопасный") from error
        return value

    @model_validator(mode="after")
    def validate_retry_bounds(self) -> Self:
        if self.retry_max_attempts > 3:
            raise ValueError("retry_max_attempts не может превышать 3")
        if self.retry_max_delay_seconds < self.retry_base_delay_seconds:
            raise ValueError("retry_max_delay_seconds не может быть меньше retry_base_delay_seconds")
        return self


class AgentRuntimeSettings(StrictFrozenModel):
    """Необязательный до wiring business agents runtime slice."""

    gateway: GatewaySettings
    default_max_turns: PositiveInt = 3
    default_max_input_tokens: PositiveInt = 8_000
    default_max_output_tokens: PositiveInt = 2_000
    default_max_cost_usd: NonNegativeDecimal = Decimal("1.00")
    default_deadline_seconds: PositiveFloat = 30.0
    max_observation_bytes: PositiveInt = 16_384

    @model_validator(mode="after")
    def validate_turn_bound(self) -> Self:
        if self.default_max_turns > 3:
            raise ValueError("default_max_turns bounded agent runtime не может превышать 3")
        if self.max_observation_bytes > 20_000:
            raise ValueError("max_observation_bytes не может превышать 20000")
        return self


class SpeechSettings(StrictFrozenModel):
    """Fail-closed граница internal speech и явно настроенного remote provider."""

    mode: SpeechTransportMode = "internal_http"
    base_url: HttpUrl
    api_key: SecretStr | None = None
    transcription_path: str = "/v1/transcribe"
    streaming_path: str = "/v1/stream"
    timeout_seconds: PositiveFloat = 180.0
    max_success_response_bytes: PositiveInt = _SPEECH_MAX_SUCCESS_RESPONSE_BYTES
    max_error_response_bytes: PositiveInt = 16 * 1024

    @field_validator("transcription_path", "streaming_path")
    @classmethod
    def validate_service_path(cls, value: str) -> str:
        return _validate_speech_service_path(value)

    @model_validator(mode="after")
    def validate_transport(self) -> Self:
        _validate_speech_base_url(self.base_url, self.mode)
        _require_speech_api_key(self.mode, self.api_key)
        _validate_speech_response_limits(self.max_success_response_bytes, self.max_error_response_bytes)
        return self


class ObservabilitySettings(StrictFrozenModel):
    """Internal-only observability settings; exporters never receive content fields."""

    enabled: bool = True
    service_name: NonEmptyId = "mtbank-ai-api"
    metrics_path: NonEmptyId = "/metrics"


class WebSocketSettings(StrictFrozenModel):
    """Bounded real-time transport settings. Disabled until a streaming adapter is provisioned."""

    enabled: bool = False
    allowed_origins: tuple[NonEmptyId, ...] = ()
    max_frame_bytes: PositiveInt = 64 * 1024
    max_session_bytes: PositiveInt = 10 * 1024 * 1024
    max_update_text_bytes: PositiveInt = 48 * 1024
    max_duration_seconds: PositiveFloat = 300.0
    max_sessions: PositiveInt = 4
    processing_timeout_seconds: PositiveFloat = 2.0

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        required_pcm_bytes = int(self.max_duration_seconds * 16_000 * 2)
        if self.max_session_bytes < self.max_frame_bytes:
            raise ValueError("max_session_bytes не может быть меньше max_frame_bytes")
        if self.max_session_bytes < required_pcm_bytes:
            raise ValueError("max_session_bytes не покрывает максимальную PCM duration")
        if self.max_update_text_bytes > self.max_frame_bytes:
            raise ValueError("max_update_text_bytes не может превышать max_frame_bytes")
        return self


class TrendsSettings(StrictFrozenModel):
    """Bounds for evidence-backed aggregate-only trend calculations."""

    enabled: bool = True
    minimum_sample_size: PositiveInt = 5
    max_window_days: PositiveInt = 90
    max_records: PositiveInt = 200

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum_sample_size < 5:
            raise ValueError("trend нельзя строить менее чем по пяти calls")
        if self.max_records > 250:
            raise ValueError("trend max_records не может превышать 250")
        if self.max_records < self.minimum_sample_size:
            raise ValueError("trend max_records не может быть меньше minimum_sample_size")
        return self


class WorkflowSettings(StrictFrozenModel):
    """Конфигурация shared deterministic analysis workflow."""

    code_sha: NonEmptyId
    provider_id: NonEmptyId = "openai-compatible"
    deadline_seconds: PositiveFloat = 60.0
    max_url_bytes: PositiveInt = 25 * 1024 * 1024
    url_timeout_seconds: PositiveFloat = 15.0
    url_max_redirects: NonNegativeInt = 3
    normalized_sample_rate_hz: PositiveInt = 16_000
    normalized_channels: PositiveInt = 1
    privacy_mode: NonEmptyId = "redacted-cloud"
    raw_audio_retention_seconds: NonNegativeInt = 0
    evidence_retention_days: NonNegativeInt = 30

    @model_validator(mode="after")
    def validate_workflow_bounds(self) -> Self:
        if self.deadline_seconds > 300:
            raise ValueError("workflow deadline_seconds не может превышать 300")
        if self.url_max_redirects > 5:
            raise ValueError("url_max_redirects не может превышать 5")
        return self


class DatabaseSettings(StrictFrozenModel):
    host: NonEmptyId = "postgres"
    port: PositiveInt = 5432
    name: NonEmptyId = "mtbank_ai"
    user: NonEmptyId = "mtbank_ai"
    password: SecretStr
    pool_size: PositiveInt = 5
    max_overflow: NonNegativeInt = 10
    pool_timeout_seconds: PositiveFloat = 2.0
    connect_timeout_seconds: PositiveFloat = 2.0
    command_timeout_seconds: PositiveFloat = 2.0
    readiness_timeout_seconds: PositiveFloat = 3.0


class DatabaseSettingsLoader(BaseSettings):
    """DB-only env projection для миграций без загрузки API-настроек."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_prefix="MTBANK_",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    database: DatabaseSettings


def load_database_settings() -> DatabaseSettings:
    return DatabaseSettingsLoader().database  # pyright: ignore[reportCallIssue]


class Settings(BaseSettings):
    """Настройки загружаются только внутри app factory."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=None,
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_prefix="MTBANK_",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    environment: NonEmptyId = "development"
    api: ApiSettings
    database: DatabaseSettings
    agent_runtime: AgentRuntimeSettings | None = None
    speech: SpeechSettings | None = None
    workflow: WorkflowSettings | None = None
    observability: ObservabilitySettings = ObservabilitySettings()
    websocket: WebSocketSettings = WebSocketSettings()
    trends: TrendsSettings = TrendsSettings()

    @model_validator(mode="after")
    def require_complete_analysis_configuration(self) -> Self:
        configured = (self.agent_runtime, self.speech, self.workflow)
        if any(value is not None for value in configured) and any(value is None for value in configured):
            raise ValueError("analysis runtime должен быть сконфигурирован целиком")
        return self
