from __future__ import annotations

import asyncio
from typing import cast

from pydantic import HttpUrl, SecretStr, TypeAdapter

from mtbank_ai.api.main import create_app
from mtbank_ai.api.readiness import CompositeReadiness, SpeechHttpReadiness
from mtbank_ai.application.ports import UnavailableAnalyzeCall, UnavailableReadiness
from mtbank_ai.config import (
    AgentRuntimeSettings,
    ApiSettings,
    DatabaseSettings,
    GatewayModelSettings,
    GatewaySettings,
    Settings,
    SpeechSettings,
    TrendsSettings,
    WebSocketSettings,
    WorkflowSettings,
)
from mtbank_ai.speech.client import HttpSpeechServiceClient
from mtbank_ai.speech.streaming import InternalSpeechWebSocketAdapter
from mtbank_ai.storage.postgres import PostgresReadiness
from mtbank_ai.trends import TrendsAgent
from mtbank_ai.workflow.analysis import AnalysisWorkflow

SAFE_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


def _settings(*, complete: bool) -> Settings:
    api = ApiSettings(api_key=SecretStr(SAFE_KEY))
    database = DatabaseSettings(password=SecretStr("opaque-database-password"))
    if not complete:
        return Settings(environment="test", api=api, database=database)
    return Settings(
        environment="test",
        api=api,
        database=database,
        agent_runtime=AgentRuntimeSettings(
            gateway=GatewaySettings(
                base_url="https://gateway.example.test/v1",
                api_key=SecretStr(SAFE_KEY),
                models=GatewayModelSettings(default_model="test-model"),
            )
        ),
        speech=SpeechSettings(base_url=TypeAdapter(HttpUrl).validate_python("http://speech:8010")),
        workflow=WorkflowSettings(code_sha="abcdef0"),
    )


def test_api_fails_closed_when_runtime_workflow_configuration_is_incomplete() -> None:
    app = create_app(settings=_settings(complete=False))

    assert isinstance(app.state.analyzer, UnavailableAnalyzeCall)
    assert isinstance(app.state.readiness, UnavailableReadiness)
    assert app.state.trends_agent is None


def test_api_builds_internal_streaming_adapter_only_when_public_websocket_is_enabled() -> None:
    disabled = create_app(settings=_settings(complete=True))
    assert disabled.state.streaming_speech is None

    settings = _settings(complete=True).model_copy(
        update={
            "websocket": WebSocketSettings(
                enabled=True,
                allowed_origins=("https://console.example.test",),
            )
        }
    )
    enabled = create_app(settings=settings)

    assert isinstance(enabled.state.streaming_speech, InternalSpeechWebSocketAdapter)
    assert enabled.state.streaming_speech._settings.url == "ws://speech:8010/v1/stream"
    assert enabled.state.streaming_speech._settings.max_message_bytes == 65_540


def test_api_builds_shared_workflow_and_readiness_only_from_complete_configuration() -> None:
    async def scenario() -> None:
        app = create_app(settings=_settings(complete=True))
        assert isinstance(app.state.analyzer, AnalysisWorkflow)
        assert isinstance(app.state.readiness, CompositeReadiness)
        assert isinstance(app.state.readiness._dependencies[0], PostgresReadiness)
        assert isinstance(app.state.readiness._dependencies[1], SpeechHttpReadiness)
        assert app.state.readiness._dependencies[1]._mode == "internal_http"
        assert app.state.readiness._dependencies[1]._api_key is None
        assert isinstance(app.state.trends_agent, TrendsAgent)
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(scenario())


def test_api_preserves_injected_readiness_seam_and_closes_it_once() -> None:
    class TrackingReadiness:
        def __init__(self) -> None:
            self.close_calls = 0

        async def ping(self) -> bool:
            return True

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        readiness = TrackingReadiness()
        app = create_app(settings=_settings(complete=True), readiness=readiness)
        assert app.state.readiness is readiness
        async with app.router.lifespan_context(app):
            pass
        assert readiness.close_calls == 1

    asyncio.run(scenario())


def test_workflow_factory_passes_typed_remote_speech_configuration_to_client() -> None:
    settings = _settings(complete=True).model_copy(
        update={
            "speech": SpeechSettings(
                mode="remote_https",
                base_url=TypeAdapter(HttpUrl).validate_python("https://speech.example.test/api"),
                api_key=SecretStr(SAFE_KEY),
                transcription_path="/v1/transcribe",
                timeout_seconds=30.0,
                max_success_response_bytes=128 * 1024,
                max_error_response_bytes=8 * 1024,
            )
        }
    )

    app = create_app(settings=settings)

    assert isinstance(app.state.analyzer, AnalysisWorkflow)
    assert isinstance(app.state.readiness, CompositeReadiness)
    speech_readiness = app.state.readiness._dependencies[1]
    assert isinstance(speech_readiness, SpeechHttpReadiness)
    assert speech_readiness._mode == "remote_https"
    assert speech_readiness._api_key == SecretStr(SAFE_KEY)
    speech_client = app.state.analyzer._speech_client
    assert isinstance(speech_client, HttpSpeechServiceClient)
    assert speech_client._settings.mode == "remote_https"
    assert speech_client._settings.base_url == TypeAdapter(HttpUrl).validate_python("https://speech.example.test/api")
    assert speech_client._settings.api_key == SecretStr(SAFE_KEY)
    assert speech_client._settings.transcription_path == "/v1/transcribe"
    assert speech_client._settings.timeout_seconds == 30.0
    assert speech_client._settings.max_success_response_bytes == 128 * 1024
    assert speech_client._settings.max_error_response_bytes == 8 * 1024


def test_api_does_not_build_trends_agent_when_feature_is_disabled() -> None:
    async def scenario() -> None:
        settings = _settings(complete=True).model_copy(update={"trends": TrendsSettings(enabled=False)})
        app = create_app(settings=settings)
        assert app.state.trends_agent is None
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(scenario())


def test_api_preserves_injected_trends_agent_and_closes_it_with_lifespan() -> None:
    class TrackingTrendsAgent:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        agent = TrackingTrendsAgent()
        app = create_app(settings=_settings(complete=False), trends_agent=cast(TrendsAgent, agent))
        assert app.state.trends_agent is agent
        async with app.router.lifespan_context(app):
            pass
        assert agent.close_calls == 1

    asyncio.run(scenario())
