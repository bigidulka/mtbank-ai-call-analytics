"""Composition root для production shared analysis workflow."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from mtbank_ai.agent_runtime import ConfiguredOpenAICompatibleGateway
from mtbank_ai.agents import CoreAgents
from mtbank_ai.config import Settings
from mtbank_ai.observability import Telemetry
from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.speech.client import HttpSpeechServiceClient, SpeechServiceClientSettings
from mtbank_ai.storage.repositories import create_sqlalchemy_trend_repository, create_sqlalchemy_uow_factory
from mtbank_ai.trends import TrendsAgent
from mtbank_ai.workflow.analysis import AnalysisWorkflow
from mtbank_ai.workflow.fetch import SafeUrlFetcher, UrlFetchPolicy


def build_configured_analysis_workflow(
    settings: Settings, *, engine: AsyncEngine, telemetry: Telemetry | None = None
) -> AnalysisWorkflow | None:
    """Строит реальный workflow только при полностью объявленных runtime settings."""

    if settings.agent_runtime is None or settings.speech is None or settings.workflow is None:
        return None

    policies = PolicyRegistry()
    policies.load_all()
    gateway = ConfiguredOpenAICompatibleGateway(settings.agent_runtime.gateway, telemetry=telemetry)
    agents = CoreAgents(
        model_client=gateway,
        runtime_settings=settings.agent_runtime,
        policies=policies,
        telemetry=telemetry,
    )
    speech_client = HttpSpeechServiceClient(
        SpeechServiceClientSettings(
            mode=settings.speech.mode,
            base_url=settings.speech.base_url,
            api_key=settings.speech.api_key,
            transcription_path=settings.speech.transcription_path,
            timeout_seconds=settings.speech.timeout_seconds,
            max_success_response_bytes=settings.speech.max_success_response_bytes,
            max_error_response_bytes=settings.speech.max_error_response_bytes,
        )
    )
    url_fetcher = SafeUrlFetcher(
        UrlFetchPolicy(
            max_bytes=settings.workflow.max_url_bytes,
            timeout_seconds=settings.workflow.url_timeout_seconds,
            max_redirects=settings.workflow.url_max_redirects,
            allowed_media_types=settings.api.allowed_media_types,
        )
    )
    return AnalysisWorkflow(
        speech_client=speech_client,
        agents=agents,
        policies=policies,
        runtime_settings=settings.agent_runtime,
        workflow_settings=settings.workflow,
        uow_factory=create_sqlalchemy_uow_factory(engine),
        url_fetcher=url_fetcher,
        telemetry=telemetry,
    )


def build_configured_trends_agent(
    settings: Settings,
    *,
    engine: AsyncEngine,
    telemetry: Telemetry | None = None,
) -> TrendsAgent | None:
    """Строит отдельный Trends LLM-agent только при полном runtime configuration."""

    if (
        not settings.trends.enabled
        or settings.agent_runtime is None
        or settings.speech is None
        or settings.workflow is None
    ):
        return None
    gateway = ConfiguredOpenAICompatibleGateway(settings.agent_runtime.gateway, telemetry=telemetry)
    return TrendsAgent(
        create_sqlalchemy_trend_repository(engine),
        settings.trends,
        model_client=gateway,
        runtime_settings=settings.agent_runtime,
        telemetry=telemetry,
    )
