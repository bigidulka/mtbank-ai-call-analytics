"""FastAPI app factory без import-time секретов и подключений."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse

from mtbank_ai.api.body_limits import BodyLimitMiddleware
from mtbank_ai.api.error_handlers import install_error_handlers
from mtbank_ai.api.readiness import CompositeReadiness, SpeechHttpReadiness
from mtbank_ai.api.routes.analyze import router as analyze_router
from mtbank_ai.api.routes.health import router as health_router
from mtbank_ai.api.routes.transcribe_ws import WebSocketSessionManager
from mtbank_ai.api.routes.transcribe_ws import router as transcribe_ws_router
from mtbank_ai.api.routes.trends import router as trends_router
from mtbank_ai.api.schemas import UrlAnalyzeRequest
from mtbank_ai.application.ports import (
    AnalyzeCallPort,
    ReadinessPort,
    UnavailableAnalyzeCall,
    UnavailableReadiness,
)
from mtbank_ai.config import Settings
from mtbank_ai.observability import Telemetry
from mtbank_ai.speech.streaming import (
    InternalSpeechWebSocketAdapter,
    InternalSpeechWebSocketSettings,
    RemoteSpeechWebSocketAdapter,
    RemoteSpeechWebSocketSettings,
    StreamingSpeechPort,
)
from mtbank_ai.storage.postgres import PostgresReadiness, create_postgres_engine
from mtbank_ai.trends import TrendsAgent
from mtbank_ai.workflow.factory import build_configured_analysis_workflow, build_configured_trends_agent

RequestHandler = Callable[[Request], Awaitable[Response]]


def create_app(
    settings: Settings | None = None,
    analyzer: AnalyzeCallPort | None = None,
    readiness: ReadinessPort | None = None,
    *,
    telemetry: Telemetry | None = None,
    streaming_speech: StreamingSpeechPort | None = None,
    trends_agent: TrendsAgent | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()  # pyright: ignore[reportCallIssue]
    resolved_telemetry = telemetry or Telemetry()
    resolved_streaming_speech = (
        streaming_speech if streaming_speech is not None else _build_streaming_speech_adapter(resolved_settings)
    )
    database_readiness: PostgresReadiness | None = None
    engine = None
    if analyzer is not None:
        resolved_analyzer = analyzer
    elif _has_complete_workflow_configuration(resolved_settings):
        engine = create_postgres_engine(resolved_settings)
        resolved_analyzer = build_configured_analysis_workflow(
            resolved_settings, engine=engine, telemetry=resolved_telemetry
        )
        if resolved_analyzer is None:
            raise RuntimeError("полная workflow configuration должна создавать AnalyzeCallUseCase")
    else:
        resolved_analyzer = UnavailableAnalyzeCall()

    if readiness is not None:
        resolved_readiness = readiness
    elif _has_complete_workflow_configuration(resolved_settings):
        speech = resolved_settings.speech
        if speech is None:
            raise RuntimeError("полная workflow configuration должна включать speech")
        database_readiness = PostgresReadiness(
            engine or create_postgres_engine(resolved_settings),
            resolved_settings.database.readiness_timeout_seconds,
        )
        speech_readiness = SpeechHttpReadiness(
            speech.base_url,
            resolved_settings.database.readiness_timeout_seconds,
            mode=speech.mode,
            api_key=speech.api_key,
        )
        resolved_readiness = CompositeReadiness(database_readiness, speech_readiness)
    elif analyzer is not None:
        database_readiness = PostgresReadiness(
            create_postgres_engine(resolved_settings),
            resolved_settings.database.readiness_timeout_seconds,
        )
        resolved_readiness = database_readiness
    else:
        resolved_readiness = UnavailableReadiness()

    resolved_trends_agent = trends_agent
    if resolved_trends_agent is None and engine is not None:
        resolved_trends_agent = build_configured_trends_agent(
            resolved_settings,
            engine=engine,
            telemetry=resolved_telemetry,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        yield
        await _close_resources(resolved_analyzer, resolved_trends_agent, resolved_readiness)

    app = FastAPI(
        title="MTBank Call Analytics API",
        version="0.1.0-foundation",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.analyzer = resolved_analyzer
    app.state.readiness = resolved_readiness
    app.state.telemetry = resolved_telemetry
    app.state.streaming_speech = resolved_streaming_speech
    app.state.trends_agent = resolved_trends_agent
    app.state.ws_sessions = WebSocketSessionManager(resolved_settings.websocket.max_sessions, resolved_telemetry)
    app.add_middleware(
        BodyLimitMiddleware,
        max_json_bytes=resolved_settings.api.max_json_bytes,
        max_upload_bytes=resolved_settings.api.max_upload_bytes,
        multipart_reserve_bytes=resolved_settings.api.multipart_reserve_bytes,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestHandler) -> Response:
        request_id = _request_id_from_header(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        with (
            resolved_telemetry.context(request_id=request_id),
            resolved_telemetry.span("http.request", method=request.method),
        ):
            response = await call_next(request)
        response.headers.setdefault("X-Request-ID", str(request_id))
        resolved_telemetry.metrics.increment(
            "mtbank_api_calls_total", status=response.status_code, route=_metric_route(request.url.path)
        )
        return response

    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(analyze_router)
    app.include_router(trends_router)
    app.include_router(transcribe_ws_router)

    @app.get(resolved_settings.observability.metrics_path, include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(resolved_telemetry.metrics.render(), media_type="text/plain; version=0.0.4")

    _install_openapi(app)
    return app


def _has_complete_workflow_configuration(settings: Settings) -> bool:
    return settings.agent_runtime is not None and settings.speech is not None and settings.workflow is not None


def _build_streaming_speech_adapter(settings: Settings) -> StreamingSpeechPort | None:
    speech = settings.speech
    websocket = settings.websocket
    if not websocket.enabled or speech is None:
        return None
    common = {
        "base_url": str(speech.base_url),
        "stream_path": speech.streaming_path,
        "open_timeout_seconds": min(websocket.processing_timeout_seconds, speech.timeout_seconds),
        "ping_interval_seconds": min(20.0, websocket.max_duration_seconds),
        "ping_timeout_seconds": websocket.processing_timeout_seconds,
        "close_timeout_seconds": 1.0,
        "max_message_bytes": websocket.max_frame_bytes + 4,
    }
    if speech.mode == "internal_http":
        return InternalSpeechWebSocketAdapter(InternalSpeechWebSocketSettings(**common))
    if speech.api_key is None:
        raise RuntimeError("remote speech configuration requires an API key")
    return RemoteSpeechWebSocketAdapter(RemoteSpeechWebSocketSettings(**common, api_key=speech.api_key))


async def _close_resources(*resources: object | None) -> None:
    closed: set[int] = set()
    for resource in resources:
        if resource is None or id(resource) in closed:
            continue
        closed.add(id(resource))
        close = getattr(resource, "close", None)
        if close is None or not callable(close):
            continue
        result: Any = close()
        if inspect.isawaitable(result):
            await result


def _request_id_from_header(value: str | None) -> UUID:
    if value is not None:
        try:
            return UUID(value)
        except ValueError:
            pass
    return uuid4()


def _metric_route(path: str) -> str:
    if path in {"/analyze", "/trends", "/metrics"}:
        return path.removeprefix("/")
    if path.startswith("/health/"):
        return "health"
    return "other"


def _install_openapi(app: FastAPI) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title,
                version=app.version,
                routes=app.routes,
            )
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            components["UrlAnalyzeRequest"] = UrlAnalyzeRequest.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
