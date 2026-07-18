from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from mtbank_ai.api.main import create_app
from mtbank_ai.config import ApiSettings, DatabaseSettings, Settings
from mtbank_ai.observability import MetricsRegistry, StructuredJsonLogger, Telemetry


class _CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_metrics_have_bounded_labels_and_redact_content() -> None:
    metrics = MetricsRegistry()
    metrics.increment("mtbank_api_calls_total", route="analyze", status=200)
    metrics.observe("mtbank_stage_latency_seconds", 0.01, stage="ingest", status="ok")
    rendered = metrics.render()
    assert 'route="analyze"' in rendered
    assert "mtbank_stage_latency_seconds_count" in rendered
    with pytest.raises(ValueError):
        metrics.increment("mtbank_bad", url="https://private.invalid/query")

    logger = _CaptureLogger()
    StructuredJsonLogger(logger).info("request", transcript="secret transcript", api_key="secret", status=200)  # type: ignore[arg-type]
    assert "secret transcript" not in logger.messages[0]
    assert "[redacted]" in logger.messages[0]


def test_metrics_endpoint_changes_after_a_real_api_call() -> None:
    async def scenario() -> None:
        app = create_app(
            settings=Settings(
                environment="test",
                api=ApiSettings(api_key=SecretStr("N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F")),
                database=DatabaseSettings(password=SecretStr("opaque-database-password")),
            )
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health/live")).status_code == 200
            metrics = (await client.get("/metrics")).text
        assert 'mtbank_api_calls_total{route="health",status="200"} 1' in metrics

    asyncio.run(scenario())


def test_telemetry_error_span_has_only_sanitized_otel_error_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry = Telemetry(tracer=provider.get_tracer("test"))
    secret = "provider response secret https://private.invalid/trace"
    with pytest.raises(RuntimeError, match="provider response"):
        with telemetry.span("agent.model_turn", prompt="private"):
            raise RuntimeError(secret)

    exported = exporter.get_finished_spans()[0]
    assert telemetry.spans[0].status == "error"
    assert telemetry.spans[0].attributes == {"error.type": "runtime", "error.code": "internal"}
    attributes = cast(Mapping[str, object], exported.attributes)
    assert attributes["error.type"] == "runtime"
    assert attributes["error.code"] == "internal"
    assert len(attributes) == 2
    assert exported.events == ()
    assert secret not in str(exported)
