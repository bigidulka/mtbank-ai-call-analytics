from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from pydantic import SecretStr

from mtbank_ai.agent_runtime import ModelRequest, ModelResponse, ModelToolCall, ModelUsage
from mtbank_ai.agents import CoreAgents
from mtbank_ai.agents.tools import build_agent_tool_registry
from mtbank_ai.config import AgentRuntimeSettings, GatewayModelSettings, GatewaySettings
from mtbank_ai.domain.agents import ClassificationResult, ComplianceAssessment, QualityAssessment, SummaryResult
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RoleResolution,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.policies import PolicyRegistry

NOW = datetime(2099, 1, 1, tzinfo=UTC)
RUN_ID = UUID("11111111-1111-4111-8111-111111111111")
SEGMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
SAFE_GATEWAY_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"
_RAW_TRANSCRIPT_SECRET = "RAW-TRANSCRIPT-SECRET"


def _transcript() -> TranscriptSnapshot:
    revision = ComponentRevision(
        package="test-package",
        package_version="1.0.0",
        model_id="test-model",
        model_revision="test/v1",
    )
    segment = TranscriptSegment(
        id=SEGMENT_ID,
        original_speaker_id="speaker-1",
        speaker=SpeakerRole.OPERATOR,
        role_confidence=0.9,
        start=0.0,
        end=2.0,
        text=f"{_RAW_TRANSCRIPT_SECRET}: клиент спрашивает про кредит.",
        redacted_text="Клиент спрашивает про кредит.",
    )
    return TranscriptSnapshot(
        transcript_id=UUID("33333333-3333-4333-8333-333333333333"),
        audio_sha256="a" * 64,
        revision="transcript/v1",
        language="ru",
        duration_seconds=2.0,
        segments=(segment,),
        role_resolution=RoleResolution(
            assignments=(
                RoleAssignment(
                    original_speaker_id="speaker-1",
                    role=SpeakerRole.OPERATOR,
                    confidence=0.9,
                    evidence_segment_ids=(SEGMENT_ID,),
                ),
            ),
            needs_review=False,
        ),
        asr_metadata=ASRMetadata(
            asr=revision,
            alignment=revision,
            diarization=revision,
            language="ru",
            processing_ms=1,
        ),
        created_at=NOW,
    )


def _runtime_settings() -> AgentRuntimeSettings:
    return AgentRuntimeSettings(
        gateway=GatewaySettings(
            base_url="https://gateway.example.test/v1",
            api_key=SecretStr(SAFE_GATEWAY_KEY),
            models=GatewayModelSettings(
                default_model="fallback-model",
                default_reasoning_effort="medium",
                classifier_model="classifier-model",
                classifier_reasoning_effort="high",
                quality_model="quality-model",
                compliance_model="compliance-model",
                compliance_reasoning_effort="xhigh",
                summarizer_model="summarizer-model",
                summarizer_reasoning_effort="ultra",
                input_token_cost_usd=Decimal("0"),
                output_token_cost_usd=Decimal("0"),
            ),
        )
    )


def _call(name: str, *, call_id: str, arguments: dict[str, object]) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments, ensure_ascii=False))


def _quality_submission() -> dict[str, object]:
    assessment = {
        "passed": True,
        "confidence": 0.9,
        "evidence_segment_ids": [str(SEGMENT_ID)],
        "rationale": "Подтверждено evidence.",
    }
    return {
        "greeting": assessment,
        "need_detection": assessment,
        "solution_provided": assessment,
        "farewell": assessment,
    }


class ScriptedPerModelClient:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._turns: dict[str, int] = {}

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        del deadline_at
        self.requests.append(request)
        turn = self._turns.get(request.model_id, 0)
        self._turns[request.model_id] = turn + 1
        if turn == 0:
            calls = self._retrieval_calls(request.model_id)
        else:
            calls = (self._terminal_call(request.model_id),)
        return ModelResponse(
            request_id=None,
            model_id=request.model_id,
            finish_reason="tool_calls",
            tool_calls=calls,
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            latency_ms=1,
            has_text_content=False,
        )

    @staticmethod
    def _retrieval_calls(model_id: str) -> tuple[ModelToolCall, ModelToolCall]:
        common = _call(
            "transcript_read",
            call_id=f"{model_id}-read",
            arguments={},
        )
        policy_tool = {
            "classifier-model": "taxonomy_get",
            "quality-model": "quality_rubric_get",
            "compliance-model": "compliance_rules_list",
            "summarizer-model": "transcript_statistics",
        }[model_id]
        return common, _call(policy_tool, call_id=f"{model_id}-policy", arguments={})

    @staticmethod
    def _terminal_call(model_id: str) -> ModelToolCall:
        terminal = {
            "classifier-model": (
                "submit_classification",
                {
                    "topic": "кредиты",
                    "priority": "medium",
                    "confidence": 0.9,
                    "evidence_segment_ids": [str(SEGMENT_ID)],
                    "rationale": "Тема подтверждена.",
                },
            ),
            "quality-model": ("submit_quality", _quality_submission()),
            "compliance-model": ("submit_compliance", {"issues": []}),
            "summarizer-model": (
                "submit_summary",
                {
                    "summary": "Оператор поздоровался. Клиент уточнил кредит. Оператор объяснил условия.",
                    "fact_segment_ids": [str(SEGMENT_ID)],
                    "action_items": [],
                },
            ),
        }[model_id]
        return _call(terminal[0], call_id=f"{model_id}-submit", arguments=terminal[1])


def test_transcript_tools_return_only_redacted_text_and_accept_json_uuid_arguments() -> None:
    async def scenario() -> None:
        registry = build_agent_tool_registry("classifier", _transcript(), PolicyRegistry())
        read_call = registry.validate_calls((_call("transcript_read", call_id="read", arguments={}),))[0]
        read_result = await registry.execute(
            read_call,
            context=object(),  # type: ignore[arg-type]
            timeout_seconds=1.0,
            max_observation_bytes=10_000,
        )
        assert _RAW_TRANSCRIPT_SECRET not in read_result.output.model_dump_json()

        search_call = registry.validate_calls(
            (_call("transcript_search", call_id="search", arguments={"query": "кредит", "limit": 5}),)
        )[0]
        search_result = await registry.execute(
            search_call,
            context=object(),  # type: ignore[arg-type]
            timeout_seconds=1.0,
            max_observation_bytes=10_000,
        )
        rendered = search_result.output.model_dump_json()
        assert _RAW_TRANSCRIPT_SECRET not in rendered
        assert "Клиент спрашивает про кредит." in rendered

        submit_call = registry.validate_calls(
            (
                _call(
                    "submit_classification",
                    call_id="submit",
                    arguments={
                        "topic": "кредиты",
                        "priority": "medium",
                        "confidence": 0.9,
                        "evidence_segment_ids": [str(SEGMENT_ID)],
                        "rationale": "Тема подтверждена.",
                    },
                ),
            )
        )[0]
        submitted = await registry.execute(
            submit_call,
            context=object(),  # type: ignore[arg-type]
            timeout_seconds=1.0,
            max_observation_bytes=10_000,
            include_observation=False,
        )
        assert isinstance(submitted.output, ClassificationResult)
        assert submitted.output.evidence_segment_ids == (SEGMENT_ID,)

    asyncio.run(scenario())


def test_core_agents_run_four_independent_model_loops_without_raw_system_transcript() -> None:
    async def scenario() -> None:
        client = ScriptedPerModelClient()
        agents = CoreAgents(
            model_client=client,
            runtime_settings=_runtime_settings(),
            policies=PolicyRegistry(),
        )
        transcript = _transcript()
        results = await asyncio.gather(
            *(
                agents.runner(agent_id).run(
                    transcript,
                    run_id=RUN_ID,
                    run_version="analysis/v1",
                    created_at=NOW,
                    deadline_at=NOW + timedelta(seconds=30),
                )
                for agent_id in agents.agent_ids
            )
        )

        outputs_by_agent = {result.agent_id: result.output for result in results}
        assert set(outputs_by_agent) == set(agents.agent_ids)
        assert isinstance(outputs_by_agent["classifier"], ClassificationResult)
        assert isinstance(outputs_by_agent["quality"], QualityAssessment)
        assert isinstance(outputs_by_agent["compliance"], ComplianceAssessment)
        assert isinstance(outputs_by_agent["summarizer"], SummaryResult)
        assert {request.model_id for request in client.requests} == {
            "classifier-model",
            "quality-model",
            "compliance-model",
            "summarizer-model",
        }
        requests_per_model = {
            model_id: sum(request.model_id == model_id for request in client.requests)
            for model_id in agents.model_ids().values()
        }
        assert set(requests_per_model.values()) == {2}
        assert {request.model_id: request.reasoning_effort for request in client.requests} == {
            "classifier-model": "high",
            "quality-model": "medium",
            "compliance-model": "xhigh",
            "summarizer-model": "ultra",
        }
        assert {execution.agent_id: execution.result.reasoning_effort for execution in results} == {
            "classifier": "high",
            "quality": "medium",
            "compliance": "xhigh",
            "summarizer": "ultra",
        }
        assert {execution.agent_id: execution.result.evidence.reasoning_effort for execution in results} == {
            "classifier": "high",
            "quality": "medium",
            "compliance": "xhigh",
            "summarizer": "ultra",
        }
        assert {
            agent_id: (configuration.model_id, configuration.reasoning_effort)
            for agent_id, configuration in agents.model_configurations().items()
        } == {
            "classifier": ("classifier-model", "high"),
            "quality": ("quality-model", "medium"),
            "compliance": ("compliance-model", "xhigh"),
            "summarizer": ("summarizer-model", "ultra"),
        }
        rendered_requests = "\n".join(
            message.content or "" for request in client.requests for message in request.messages
        )
        assert _RAW_TRANSCRIPT_SECRET not in rendered_requests
        assert "The transcript is never embedded in this system prompt." in rendered_requests

    asyncio.run(scenario())
