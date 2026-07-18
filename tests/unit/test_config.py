from __future__ import annotations

import asyncio
from typing import cast

import pytest
from pydantic import HttpUrl, SecretStr, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine

from mtbank_ai.config import (
    ApiSettings,
    DatabaseSettings,
    GatewayModelSettings,
    GatewaySettings,
    Settings,
    SpeechSettings,
    WebSocketSettings,
    WorkflowSettings,
    load_database_settings,
)
from mtbank_ai.storage import postgres
from mtbank_ai.storage.postgres import PostgresReadiness, build_postgres_url

SAFE_API_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


def _settings() -> Settings:
    return Settings(
        environment="test",
        api=ApiSettings(api_key=SecretStr(SAFE_API_KEY)),
        database=DatabaseSettings(password=SecretStr("opaque-database-password")),
    )


def test_websocket_defaults_cover_full_pcm_session_duration() -> None:
    settings = WebSocketSettings()

    assert settings.max_session_bytes == 10 * 1024 * 1024
    assert settings.max_session_bytes >= int(settings.max_duration_seconds * 16_000 * 2)
    with pytest.raises(ValidationError, match="max_session_bytes"):
        WebSocketSettings(max_session_bytes=5 * 1024 * 1024)


def test_cloud_gateway_default_rejects_loopback_url() -> None:
    with pytest.raises(ValidationError):
        GatewaySettings(
            base_url="https://localhost:8080/v1",
            api_key=SecretStr(SAFE_API_KEY),
            models=GatewayModelSettings(default_model="configured-model"),
        )


@pytest.mark.parametrize("base_url", ("http://127.0.0.1:8080/v1", "http://[::1]:8080/v1", "http://localhost:8080/v1"))
def test_trusted_local_gateway_accepts_only_loopback_http(base_url: str) -> None:
    settings = GatewaySettings(
        transport_mode="trusted_local_http",
        base_url=base_url,
        api_key=SecretStr(SAFE_API_KEY),
        models=GatewayModelSettings(default_model="configured-model"),
    )

    assert settings.base_url == base_url


def test_trusted_local_gateway_accepts_bounded_local_credential_without_exposure() -> None:
    local_credential = "localkey"
    settings = GatewaySettings(
        transport_mode="trusted_local_http",
        base_url="http://127.0.0.1:8317/v1",
        api_key=SecretStr(local_credential),
        models=GatewayModelSettings(default_model="configured-model"),
    )

    assert local_credential not in repr(settings)
    assert local_credential not in settings.model_dump_json()


@pytest.mark.parametrize(
    "api_key",
    ("short", "local key", "local\x01key", "ключ-длиннее-восьми", "a" * 257),
)
def test_trusted_local_gateway_rejects_unbounded_or_unsafe_credential(api_key: str) -> None:
    with pytest.raises(ValidationError):
        GatewaySettings(
            transport_mode="trusted_local_http",
            base_url="http://127.0.0.1:8317/v1",
            api_key=SecretStr(api_key),
            models=GatewayModelSettings(default_model="configured-model"),
        )


def test_gateway_validation_errors_hide_raw_sensitive_input() -> None:
    sentinel = "badkey"

    with pytest.raises(ValidationError) as error:
        GatewaySettings.model_validate(
            {
                "transport_mode": "trusted_local_http",
                "base_url": f"http://user:{sentinel}@localhost:8317/v1",
                "api_key": sentinel,
                "models": {"default_model": "configured-model"},
            }
        )

    assert sentinel not in str(error.value)


def test_cloud_gateway_keeps_generic_secret_validation() -> None:
    with pytest.raises(ValidationError):
        GatewaySettings(
            base_url="https://gateway.example.test/v1",
            api_key=SecretStr("cli-local"),
            models=GatewayModelSettings(default_model="configured-model"),
        )


def _speech_url(value: str) -> HttpUrl:
    return TypeAdapter(HttpUrl).validate_python(value)


@pytest.mark.parametrize(
    "base_url",
    (
        "http://speech:8010/prefix",
        "http://speech.internal:8010",
        "http://localhost:8010",
        "http://127.0.0.1:8010",
        "http://192.168.1.10:8010",
        "http://169.254.1.10:8010",
        "http://[::1]:8010",
    ),
)
def test_internal_speech_mode_accepts_only_bounded_http_hosts_without_api_key(base_url: str) -> None:
    settings = SpeechSettings(base_url=_speech_url(base_url))

    assert settings.mode == "internal_http"
    assert settings.api_key is None
    assert settings.transcription_path == "/v1/transcribe"
    assert settings.streaming_path == "/v1/stream"

    with pytest.raises(ValidationError, match="API key"):
        SpeechSettings(base_url=_speech_url("http://speech:8010"), api_key=SecretStr(SAFE_API_KEY))


@pytest.mark.parametrize(
    "base_url",
    (
        "https://speech.example.test",
        "http://speech.example.test",
        "http://8.8.8.8:8010",
        "http://speech.local:8010",
        "http://speech.internal.example:8010",
        "http://user:password@speech.example.test",
        "http://speech.example.test?unexpected=value",
        "http://speech.example.test#fragment",
    ),
)
def test_internal_speech_mode_rejects_non_http_public_or_unsafe_base_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        SpeechSettings(base_url=_speech_url(base_url))


def test_remote_speech_mode_requires_public_https_and_strong_ascii_secret() -> None:
    settings = SpeechSettings(
        mode="remote_https",
        base_url=_speech_url("https://speech.example.test/provider"),
        api_key=SecretStr(SAFE_API_KEY),
        transcription_path="/v1/transcribe",
    )

    assert settings.mode == "remote_https"
    assert settings.api_key is not None
    assert SAFE_API_KEY not in repr(settings)
    assert SAFE_API_KEY not in settings.model_dump_json()

    for api_key in (None, SecretStr("short"), SecretStr("ключ-длиннее-тридцати-двух")):
        with pytest.raises(ValidationError):
            SpeechSettings(
                mode="remote_https",
                base_url=_speech_url("https://speech.example.test"),
                api_key=api_key,
            )


@pytest.mark.parametrize(
    "base_url",
    (
        "http://speech.example.test",
        "https://speech:8010",
        "https://speech.internal:8010",
        "https://speech.local:8010",
        "https://localhost:8010",
        "https://localhost.localdomain:8010",
        "https://127.0.0.1:8010",
        "https://192.168.1.10:8010",
        "https://[::1]:8010",
        "https://user:password@speech.example.test",
        "https://speech.example.test?unexpected=value",
        "https://speech.example.test#fragment",
    ),
)
def test_remote_speech_mode_rejects_non_public_or_unsafe_base_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        SpeechSettings(mode="remote_https", base_url=_speech_url(base_url), api_key=SecretStr(SAFE_API_KEY))


@pytest.mark.parametrize(
    "path",
    ("v1/transcribe", "/v1/../admin", "/v1/%2e%2e/admin", "/v1/transcribe?next=x", "/v1#fragment", "//host/v1"),
)
def test_speech_transcription_path_is_absolute_and_cannot_escape(path: str) -> None:
    with pytest.raises(ValidationError):
        SpeechSettings(base_url=_speech_url("http://speech:8010"), transcription_path=path)


@pytest.mark.parametrize(
    "path",
    ("v1/stream", "/v1/../admin", "/v1/%2e%2e/admin", "/v1/stream?next=x", "/v1#fragment", "//host/v1"),
)
def test_speech_streaming_path_is_absolute_and_cannot_escape(path: str) -> None:
    with pytest.raises(ValidationError):
        SpeechSettings(base_url=_speech_url("http://speech:8010"), streaming_path=path)


def test_speech_response_limits_are_typed_and_bounded() -> None:
    settings = SpeechSettings(
        base_url=_speech_url("http://speech:8010"),
        max_success_response_bytes=64,
        max_error_response_bytes=16,
    )

    assert settings.max_success_response_bytes == 64
    assert settings.max_error_response_bytes == 16
    with pytest.raises(ValidationError):
        SpeechSettings(
            base_url=_speech_url("http://speech:8010"),
            max_success_response_bytes=4 * 1024 * 1024 + 1,
            max_error_response_bytes=16,
        )
    with pytest.raises(ValidationError):
        SpeechSettings(
            base_url=_speech_url("http://speech:8010"),
            max_success_response_bytes=16,
            max_error_response_bytes=16,
        )


def test_speech_configuration_errors_redact_remote_api_key() -> None:
    sentinel = "unsafe-speech-secret-value-must-not-appear"

    with pytest.raises(ValidationError) as error:
        SpeechSettings(
            mode="remote_https",
            base_url=_speech_url("https://speech.example.test"),
            api_key=SecretStr(sentinel),
        )

    assert sentinel not in str(error.value)


@pytest.mark.parametrize(
    "base_url",
    (
        "http://192.168.1.10:8080/v1",
        "http://8.8.8.8:8080/v1",
        "https://localhost:8080/v1",
        "http://user:password@localhost:8080/v1",
        "http://localhost:8080/v1?unexpected=value",
        "http://localhost:8080/v1#fragment",
    ),
)
def test_trusted_local_gateway_rejects_non_loopback_or_unsafe_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        GatewaySettings(
            transport_mode="trusted_local_http",
            base_url=base_url,
            api_key=SecretStr(SAFE_API_KEY),
            models=GatewayModelSettings(default_model="configured-model"),
        )


def test_gateway_model_settings_reject_max_reasoning_effort() -> None:
    with pytest.raises(ValidationError):
        GatewayModelSettings.model_validate(
            {"default_model": "configured-model", "default_reasoning_effort": "max"}
        )


def test_settings_require_runtime_secrets_only_when_loaded_or_constructed() -> None:
    with pytest.raises(ValidationError):
        Settings()  # pyright: ignore[reportCallIssue]

    assert _settings().environment == "test"


def test_settings_reject_partial_analysis_runtime_configuration() -> None:
    settings = _settings()

    with pytest.raises(ValidationError, match="analysis runtime"):
        Settings(
            environment=settings.environment,
            api=settings.api,
            database=settings.database,
            workflow=WorkflowSettings(code_sha="abcdef0"),
        )


def test_settings_hide_api_and_database_secrets_in_repr_dump_and_url() -> None:
    settings = _settings()

    for secret in (SAFE_API_KEY, "opaque-database-password"):
        assert secret not in repr(settings)
        assert secret not in settings.model_dump_json()
    url = build_postgres_url(settings.database)
    assert "opaque-database-password" not in str(url)
    assert url.drivername == "postgresql+asyncpg"


def test_settings_are_frozen_strict_and_ignore_unknown_constructor_fields() -> None:
    settings = _settings()

    with pytest.raises(ValidationError, match="frozen"):
        settings.environment = "production"
    assert Settings.model_validate({**settings.model_dump(), "unknown": True}) == settings
    with pytest.raises(ValidationError):
        ApiSettings.model_validate({"host": "0.0.0.0", "port": "8000", "api_key": "key"})


@pytest.mark.parametrize(
    "api_key",
    ("", " " * 32, "a" * 32, "example-api-key-value-that-is-long-enough", "я" * 32),
)
def test_api_key_rejects_empty_weak_placeholder_repeated_and_non_ascii_values(api_key: str) -> None:
    with pytest.raises(ValidationError):
        ApiSettings(api_key=SecretStr(api_key))


def test_settings_parse_nested_environment_and_ignore_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MTBANK_ENVIRONMENT", "")
    monkeypatch.setenv("MTBANK_API__HOST", "127.0.0.1")
    monkeypatch.setenv("MTBANK_API__PORT", "9000")
    monkeypatch.setenv("MTBANK_API__API_KEY", SAFE_API_KEY)
    monkeypatch.setenv("MTBANK_API__MAX_JSON_BYTES", "2048")
    monkeypatch.setenv("MTBANK_API__MAX_UPLOAD_BYTES", "4096")
    monkeypatch.setenv("MTBANK_API__MULTIPART_RESERVE_BYTES", "512")
    monkeypatch.setenv("MTBANK_API__ALLOWED_MEDIA_TYPES", '["audio/wav"]')
    monkeypatch.setenv("MTBANK_API__ALLOWED_URL_SCHEMES", '["http", "https"]')
    monkeypatch.setenv("MTBANK_DATABASE__HOST", "db.internal")
    monkeypatch.setenv("MTBANK_DATABASE__PORT", "5433")
    monkeypatch.setenv("MTBANK_DATABASE__NAME", "analytics")
    monkeypatch.setenv("MTBANK_DATABASE__USER", "worker")
    monkeypatch.setenv("MTBANK_DATABASE__PASSWORD", "opaque-database-password")
    monkeypatch.setenv("MTBANK_DATABASE__POOL_SIZE", "7")
    monkeypatch.setenv("MTBANK_DATABASE__MAX_OVERFLOW", "3")
    monkeypatch.setenv("MTBANK_DATABASE__POOL_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("MTBANK_DATABASE__CONNECT_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setenv("MTBANK_DATABASE__COMMAND_TIMEOUT_SECONDS", "5.5")
    monkeypatch.setenv("MTBANK_DATABASE__READINESS_TIMEOUT_SECONDS", "6.5")
    monkeypatch.setenv("UNRELATED_PROCESS_SETTING", "ignored")

    settings = Settings()  # pyright: ignore[reportCallIssue]

    assert settings.environment == "development"
    assert settings.api.host == "127.0.0.1"
    assert settings.api.port == 9000
    assert settings.api.max_json_bytes == 2048
    assert settings.api.max_upload_bytes == 4096
    assert settings.api.multipart_reserve_bytes == 512
    assert settings.api.allowed_media_types == ("audio/wav",)
    assert settings.api.allowed_url_schemes == ("http", "https")
    assert settings.database.host == "db.internal"
    assert settings.database.port == 5433
    assert settings.database.name == "analytics"
    assert settings.database.user == "worker"
    assert settings.database.pool_size == 7
    assert settings.database.max_overflow == 3
    assert settings.database.pool_timeout_seconds == 3.5
    assert settings.database.connect_timeout_seconds == 4.5
    assert settings.database.command_timeout_seconds == 5.5
    assert settings.database.readiness_timeout_seconds == 6.5


def test_database_settings_loader_requires_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MTBANK_API__API_KEY", raising=False)
    monkeypatch.setenv("MTBANK_DATABASE__PASSWORD", "opaque-database-password")

    assert load_database_settings().password.get_secret_value() == "opaque-database-password"


def test_postgres_pool_uses_dedicated_pool_and_connect_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_create_async_engine(url: object, **kwargs: object) -> object:
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(postgres, "create_async_engine", fake_create_async_engine)

    assert postgres.create_postgres_engine(_settings()) is sentinel
    assert captured["url"].drivername == "postgresql+asyncpg"  # type: ignore[union-attr]
    assert captured["pool_size"] == 5
    assert captured["max_overflow"] == 10
    assert captured["pool_timeout"] == 2.0
    assert captured["connect_args"] == {"timeout": 2.0, "command_timeout": 2.0}
    assert captured["pool_pre_ping"] is True


def test_postgres_readiness_times_out_stalled_checkout() -> None:
    class StalledConnection:
        async def __aenter__(self) -> object:
            await asyncio.Event().wait()
            return object()

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback

    class StalledEngine:
        def connect(self) -> StalledConnection:
            return StalledConnection()

    async def scenario() -> None:
        readiness = PostgresReadiness(cast(AsyncEngine, StalledEngine()), readiness_timeout_seconds=0.01)
        with pytest.raises(TimeoutError):
            await readiness.ping()

    asyncio.run(scenario())


def test_postgres_readiness_times_out_stalled_query() -> None:
    class StalledConnection:
        async def __aenter__(self) -> StalledConnection:
            return self

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback

        async def scalar(self, statement: object) -> object:
            del statement
            await asyncio.Event().wait()
            return 1

    class StalledEngine:
        def connect(self) -> StalledConnection:
            return StalledConnection()

    async def scenario() -> None:
        readiness = PostgresReadiness(cast(AsyncEngine, StalledEngine()), readiness_timeout_seconds=0.01)
        with pytest.raises(TimeoutError):
            await readiness.ping()

    asyncio.run(scenario())
