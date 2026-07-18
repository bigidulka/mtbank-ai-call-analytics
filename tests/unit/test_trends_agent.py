from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from mtbank_ai.agent_runtime import (
    AgentFailureCode,
    AgentRuntimeError,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
)
from mtbank_ai.api.main import create_app
from mtbank_ai.config import (
    AgentRuntimeSettings,
    ApiSettings,
    DatabaseSettings,
    GatewayModelSettings,
    GatewaySettings,
    Settings,
    TrendsSettings,
)
from mtbank_ai.domain.analysis import AnalysisVersions, SanitizedAnalysisRecord, SanitizedQualityChecklist
from mtbank_ai.domain.base import ReasoningEffort
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.trends import InMemoryTrendRepository, TrendRejected, TrendRequest, TrendsAgent

NOW = datetime(2026, 7, 18, tzinfo=UTC)
SAFE_API_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"
TREND_RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


def test_trends_settings_bound_record_count_to_a_feasible_limit() -> None:
    assert TrendsSettings().max_records == 200
    assert TrendsSettings(max_records=5).max_records == 5
    with pytest.raises(ValueError, match="250"):
        TrendsSettings(max_records=251)
    with pytest.raises(ValueError, match="minimum_sample_size"):
        TrendsSettings(max_records=4)


class ScriptedTrendClient:
    def __init__(self, responses: Sequence[ModelResponse | BaseException]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []
        self.closed = False

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        del deadline_at
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True


def _runtime_settings(
    *,
    trends_model: str | None = "trends-model",
    trends_reasoning_effort: ReasoningEffort | None = "high",
) -> AgentRuntimeSettings:
    return AgentRuntimeSettings(
        gateway=GatewaySettings(
            transport_mode="trusted_local_http",
            base_url="http://127.0.0.1:8317/v1",
            api_key=SecretStr("localkey"),
            models=GatewayModelSettings(
                default_model="default-model",
                default_reasoning_effort="medium",
                trends_model=trends_model,
                trends_reasoning_effort=trends_reasoning_effort,
                input_token_cost_usd=Decimal("0.001"),
                output_token_cost_usd=Decimal("0.002"),
            ),
        ),
        default_deadline_seconds=30.0,
    )


def _record(run_id: int, topic: str) -> SanitizedAnalysisRecord:
    revision = ComponentRevision(package="test", package_version="1", model_id="model", model_revision="v1")
    return SanitizedAnalysisRecord(
        run_id=UUID(int=run_id),
        classification_topic_id=topic,
        classification_priority_id="medium",
        classification_confidence=0.9,
        quality_total=80.0,
        quality_checklist=SanitizedQualityChecklist(
            greeting=True,
            need_detection=True,
            solution_provided=True,
            farewell=True,
        ),
        compliance_passed=True,
        compliance_issues=(),
        action_item_count=1,
        needs_review=False,
        processing_ms=1,
        trusted_versions=AnalysisVersions(
            code_sha="abcdef0",
            prompt_bundle_hash="a" * 64,
            taxonomy_version="taxonomy/v1",
            quality_rubric_version="quality/v1",
            compliance_policy_version="compliance/v1",
            asr=revision,
            alignment=revision,
            diarization=revision,
        ),
    )


_DEFAULT_RECORDS: tuple[tuple[int, str], ...] = (
    (1, "credit"),
    (2, "credit"),
    (3, "cards"),
    (4, "credit"),
    (5, "cards"),
)


def _repository(records: Sequence[tuple[int, str]] = _DEFAULT_RECORDS) -> InMemoryTrendRepository:
    repository = InMemoryTrendRepository()
    for run_id, topic in records:
        repository.add(_record(run_id, topic), created_at=NOW)
    return repository


def _call(name: str, call_id: str, arguments: dict[str, object]) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments, ensure_ascii=False))


def _response(*calls: ModelToolCall, model_id: str = "trends-model") -> ModelResponse:
    return ModelResponse(
        request_id=None,
        model_id=model_id,
        finish_reason="tool_calls",
        tool_calls=calls,
        usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        latency_ms=1,
        has_text_content=False,
    )


def _submission(*, supporting_run_ids: tuple[UUID, ...] = (UUID(int=1), UUID(int=2), UUID(int=4))) -> dict[str, object]:
    return {
        "qualitative_pattern": "Кредитные обращения преобладают в выбранном окне.",
        "confidence": 0.8,
        "recommendation": "Проверить причины роста кредитных обращений.",
        "supporting_run_ids": [str(run_id) for run_id in supporting_run_ids],
    }


def _client_for_success(*, model_id: str = "trends-model") -> ScriptedTrendClient:
    return ScriptedTrendClient(
        (
            _response(
                _call("trend_aggregate_query", "aggregate", {}),
                _call("trend_evidence_retrieve", "evidence", {}),
                model_id=model_id,
            ),
            _response(_call("submit_trend", "submit", _submission()), model_id=model_id),
        )
    )


def _agent(
    repository: InMemoryTrendRepository,
    client: ScriptedTrendClient,
    *,
    settings: TrendsSettings | None = None,
    runtime_settings: AgentRuntimeSettings | None = None,
) -> TrendsAgent:
    return TrendsAgent(
        repository,
        settings or TrendsSettings(),
        model_client=client,
        runtime_settings=runtime_settings or _runtime_settings(),
        now=lambda: NOW,
        run_id_factory=lambda: TREND_RUN_ID,
    )


def _request() -> TrendRequest:
    return TrendRequest(
        window_start=NOW - timedelta(minutes=1),
        window_end=NOW + timedelta(minutes=1),
        topic="credit",
    )


def test_trends_agent_requires_both_aggregate_tools_and_preserves_model_provenance() -> None:
    async def scenario() -> None:
        client = _client_for_success()
        result = await _agent(_repository(), client).analyze(_request())

        assert (result.numerator, result.denominator, result.rate) == (3, 5, 0.6)
        assert result.evidence.run_ids == (UUID(int=1), UUID(int=2), UUID(int=4))
        assert result.supporting_run_ids == result.evidence.run_ids
        assert result.qualitative_pattern == "Кредитные обращения преобладают в выбранном окне."
        assert result.agent_evidence.model_id == "trends-model"
        assert result.agent_evidence.reasoning_effort == "high"
        assert result.agent_evidence.prompt.prompt_id == "trends"
        assert len(result.agent_evidence.prompt.bundle_hash) == 64
        assert result.agent_evidence.usage == ModelUsage(input_tokens=6, output_tokens=4, total_tokens=10)
        assert result.agent_evidence.cost_usd == Decimal("0.014")
        assert len(result.agent_evidence.trajectory) > 0
        assert len(client.requests) == 2
        assert tuple(tool.name for tool in client.requests[0].tools) == (
            "trend_aggregate_query",
            "trend_evidence_retrieve",
        )
        assert tuple(tool.name for tool in client.requests[1].tools) == ("submit_trend",)
        assert {request.model_id for request in client.requests} == {"trends-model"}
        assert {request.reasoning_effort for request in client.requests} == {"high"}
        assert client.requests[1].messages[-1].role.value == "tool"
        assert "untrusted_tool_result" in (client.requests[1].messages[-1].content or "")
        rendered_provenance = result.agent_evidence.model_dump_json()
        assert "Кредитные обращения" not in rendered_provenance
        assert "Проверить причины" not in rendered_provenance

    asyncio.run(scenario())


def test_trends_agent_falls_back_to_default_model_and_effort() -> None:
    async def scenario() -> None:
        runtime_settings = _runtime_settings(trends_model=None, trends_reasoning_effort=None)
        client = _client_for_success(model_id="default-model")
        result = await _agent(_repository(), client, runtime_settings=runtime_settings).analyze(_request())

        assert result.agent_evidence.model_id == "default-model"
        assert result.agent_evidence.reasoning_effort == "medium"
        assert {request.model_id for request in client.requests} == {"default-model"}
        assert {request.reasoning_effort for request in client.requests} == {"medium"}

    asyncio.run(scenario())


def test_trends_preflight_rejects_invalid_samples_before_any_model_call() -> None:
    async def scenario() -> None:
        cases = (
            (
                _repository(((1, "credit"), (1, "credit"), (2, "credit"), (3, "cards"), (4, "cards"))),
                TrendsSettings(),
            ),
            (_repository(((1, "credit"), (2, "credit"), (3, "cards"), (4, "cards"))), TrendsSettings()),
            (_repository(()), TrendsSettings()),
            (
                _repository(
                    ((1, "credit"), (2, "credit"), (3, "cards"), (4, "credit"), (5, "cards"), (6, "credit"))
                ),
                TrendsSettings(max_records=5),
            ),
        )
        for repository, settings in cases:
            client = ScriptedTrendClient(())
            with pytest.raises(TrendRejected):
                await _agent(repository, client, settings=settings).analyze(_request())
            assert client.requests == []

        client = ScriptedTrendClient(())
        with pytest.raises(TrendRejected):
            await _agent(_repository(), client).analyze(
                TrendRequest(
                    window_start=NOW - timedelta(days=91),
                    window_end=NOW,
                    topic="credit",
                )
            )
        assert client.requests == []

        client = ScriptedTrendClient(())
        with pytest.raises(TrendRejected):
            await _agent(_repository(), client).analyze(
                TrendRequest(
                    window_start=NOW - timedelta(minutes=1),
                    window_end=NOW + timedelta(minutes=1),
                    topic="complaints",
                )
            )
        assert client.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("terminal_arguments", "expected_code"),
    (
        (
            {
                **_submission(),
                "numerator": 0,
            },
            AgentFailureCode.TOOL_ARGUMENTS_INVALID,
        ),
        (
            _submission(supporting_run_ids=(UUID(int=1), UUID(int=2))),
            AgentFailureCode.TERMINAL_SUBMIT_INVALID,
        ),
    ),
)
def test_trends_agent_rejects_model_math_override_and_mismatched_evidence(
    terminal_arguments: dict[str, object],
    expected_code: AgentFailureCode,
) -> None:
    async def scenario() -> None:
        client = ScriptedTrendClient(
            (
                _response(
                    _call("trend_aggregate_query", "aggregate", {}),
                    _call("trend_evidence_retrieve", "evidence", {}),
                ),
                _response(_call("submit_trend", "submit", terminal_arguments)),
            )
        )
        with pytest.raises(AgentRuntimeError) as error:
            await _agent(_repository(), client).analyze(_request())
        assert error.value.code is expected_code

    asyncio.run(scenario())


def test_trends_agent_rejects_missing_retrieval_and_provider_model_mismatch() -> None:
    async def scenario() -> None:
        missing = ScriptedTrendClient((_response(_call("submit_trend", "submit", _submission())),))
        with pytest.raises(AgentRuntimeError) as missing_error:
            await _agent(_repository(), missing).analyze(_request())
        assert missing_error.value.code is AgentFailureCode.REQUIRED_RETRIEVAL_MISSING

        mismatch = ScriptedTrendClient(
            (_response(_call("trend_aggregate_query", "aggregate", {}), model_id="other-model"),)
        )
        with pytest.raises(AgentRuntimeError) as mismatch_error:
            await _agent(_repository(), mismatch).analyze(_request())
        assert mismatch_error.value.code is AgentFailureCode.MODEL_MISMATCH

    asyncio.run(scenario())


def test_trends_api_serializes_the_bounded_agent_result() -> None:
    async def scenario() -> None:
        client = _client_for_success()
        agent = _agent(_repository(), client)
        settings = Settings(
            environment="test",
            api=ApiSettings(api_key=SecretStr(SAFE_API_KEY)),
            database=DatabaseSettings(password=SecretStr("opaque-database-password")),
        )
        app = create_app(settings=settings, trends_agent=agent)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as transport:
            response = await transport.post(
                "/trends",
                headers={"Authorization": f"Bearer {SAFE_API_KEY}"},
                json={
                    "window_start": "2026-07-17T23:59:00Z",
                    "window_end": "2026-07-18T00:01:00Z",
                    "topic": "credit",
                },
            )

        assert response.status_code == 200
        payload = response.json()
        assert payload["numerator"] == 3
        assert payload["denominator"] == 5
        assert payload["supporting_run_ids"] == [str(UUID(int=1)), str(UUID(int=2)), str(UUID(int=4))]
        assert payload["agent_evidence"]["model_id"] == "trends-model"
        assert "Do not request transcripts" not in response.text

    asyncio.run(scenario())


def test_trends_api_maps_preflight_and_provider_failures_to_typed_errors() -> None:
    class ProviderFailingAgent:
        async def analyze(self, request: TrendRequest):  # type: ignore[no-untyped-def]
            del request
            raise AgentRuntimeError(AgentFailureCode.PROVIDER_SERVER)

    async def scenario() -> None:
        settings = Settings(
            environment="test",
            api=ApiSettings(api_key=SecretStr(SAFE_API_KEY)),
            database=DatabaseSettings(password=SecretStr("opaque-database-password")),
        )
        app = create_app(settings=settings, trends_agent=cast(TrendsAgent, ProviderFailingAgent()))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/trends",
                headers={"Authorization": f"Bearer {SAFE_API_KEY}"},
                json={
                    "window_start": "2026-07-18T00:00:00Z",
                    "window_end": "2026-07-18T01:00:00Z",
                    "topic": "credit",
                },
            )

        assert response.status_code == 502
        assert response.json()["error"]["code"] == "provider_failure"

    asyncio.run(scenario())


def test_trends_agent_closes_its_own_model_client() -> None:
    async def scenario() -> None:
        client = ScriptedTrendClient(())
        agent = _agent(_repository(), client)
        await agent.close()
        assert client.closed is True

    asyncio.run(scenario())
