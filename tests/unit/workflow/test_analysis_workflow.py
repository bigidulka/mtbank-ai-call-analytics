from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from pydantic import SecretStr

from mtbank_ai.agent_runtime import AgentFailureCode, AgentRuntimeError
from mtbank_ai.application.ports import FileAnalyzeInput
from mtbank_ai.config import AgentRuntimeSettings, GatewayModelSettings, GatewaySettings, WorkflowSettings
from mtbank_ai.domain.agents import (
    ActionItem,
    ClassificationResult,
    ComplianceAssessment,
    ComplianceIssue,
    ComplianceSeverity,
    QualityAssessment,
    QualityCriterionAssessment,
    SummaryResult,
)
from mtbank_ai.domain.analysis import SanitizedAnalysisRecord
from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.domain.events import RunEvent, RunStatus
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RoleResolution,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.evidence.envelope import RunEnvelope
from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.speech.contracts import SpeechTranscriptionResponse
from mtbank_ai.workflow.analysis import AnalysisWorkflow, UnitOfWorkFactory

NOW = datetime(2026, 7, 16, tzinfo=UTC)
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
    return TranscriptSnapshot(
        transcript_id=UUID("33333333-3333-4333-8333-333333333333"),
        audio_sha256="a" * 64,
        revision="transcript/v1",
        language="ru",
        duration_seconds=3.0,
        segments=(
            TranscriptSegment(
                id=SEGMENT_ID,
                original_speaker_id="speaker-1",
                speaker=SpeakerRole.OPERATOR,
                role_confidence=0.9,
                start=0.0,
                end=3.0,
                text=f"{_RAW_TRANSCRIPT_SECRET}: клиент спрашивает про кредит.",
                redacted_text="Клиент спрашивает про кредит.",
            ),
        ),
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
            models=GatewayModelSettings(default_model="fallback-model"),
        )
    )


def _outputs() -> dict[str, object]:
    assessment = QualityCriterionAssessment(
        passed=True,
        confidence=0.9,
        evidence_segment_ids=(SEGMENT_ID,),
        rationale="Критерий подтверждён.",
    )
    farewell = assessment.model_copy(update={"passed": False})
    return {
        "classifier": ClassificationResult(
            topic="кредиты",
            priority="medium",
            confidence=0.9,
            evidence_segment_ids=(SEGMENT_ID,),
            rationale="Тема подтверждена.",
        ),
        "quality": QualityAssessment(
            greeting=assessment,
            need_detection=assessment,
            solution_provided=assessment,
            farewell=farewell,
        ),
        "compliance": ComplianceAssessment(
            issues=(
                ComplianceIssue(
                    rule_id="no_unconditional_guarantee",
                    severity=ComplianceSeverity.BLOCKING,
                    evidence_segment_ids=(SEGMENT_ID,),
                    explanation="Нарушение зафиксировано.",
                ),
            )
        ),
        "summarizer": SummaryResult(
            summary="Оператор поздоровался. Клиент уточнил кредит. Оператор объяснил дальнейшие действия.",
            fact_segment_ids=(SEGMENT_ID,),
            action_items=(ActionItem(text="Отправить условия.", evidence_segment_ids=(SEGMENT_ID,)),),
        ),
    }


class FakeSpeechClient:
    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self._delay_seconds = delay_seconds

    async def transcribe(self, source: object) -> SpeechTranscriptionResponse:
        del source
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        return SpeechTranscriptionResponse(transcript=_transcript())


class FakeRunner:
    def __init__(self, output: object) -> None:
        self._output = output

    async def run(self, transcript: TranscriptSnapshot, **kwargs: object) -> SimpleNamespace:
        del transcript, kwargs
        if isinstance(self._output, BaseException):
            raise self._output
        await asyncio.sleep(0)
        return SimpleNamespace(output=self._output)


class FakeAgents:
    agent_ids = ("classifier", "quality", "compliance", "summarizer")

    def __init__(self, outputs: Mapping[str, object]) -> None:
        self._runners = {agent_id: FakeRunner(outputs[agent_id]) for agent_id in self.agent_ids}

    def runner(self, agent_id: str) -> FakeRunner:
        return self._runners[agent_id]

    def model_ids(self) -> Mapping[str, str]:
        return {agent_id: f"{agent_id}-model" for agent_id in self.agent_ids}

    def model_configurations(self) -> Mapping[str, SimpleNamespace]:
        reasoning_efforts = {
            "classifier": "high",
            "quality": "medium",
            "compliance": None,
            "summarizer": "ultra",
        }
        return {
            agent_id: SimpleNamespace(
                model_id=f"{agent_id}-model",
                reasoning_effort=reasoning_efforts[agent_id],
            )
            for agent_id in self.agent_ids
        }

    def prompt_bundle_hash(self, transcript: TranscriptSnapshot) -> str:
        del transcript
        return "b" * 64


class MemoryRuns:
    def __init__(self) -> None:
        self.envelope: RunEnvelope | None = None
        self.statuses: list[tuple[RunStatus, ErrorCode | None]] = []

    async def create(self, envelope: RunEnvelope) -> None:
        self.envelope = envelope

    async def get(self, run_id: UUID) -> RunEnvelope | None:
        return self.envelope if self.envelope is not None and self.envelope.run_id == run_id else None

    async def set_status(self, run_id: UUID, status: RunStatus, *, error_code: ErrorCode | None = None) -> None:
        assert self.envelope is not None and self.envelope.run_id == run_id
        self.statuses.append((status, error_code))


class MemoryEvents:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append(self, event: RunEvent) -> None:
        self.events.append(event)

    async def list(self, run_id: UUID) -> tuple[RunEvent, ...]:
        return tuple(event for event in self.events if event.run_id == run_id)


class MemoryAnalyses:
    def __init__(self) -> None:
        self.records: list[SanitizedAnalysisRecord] = []

    async def save_sanitized(self, record: SanitizedAnalysisRecord) -> None:
        self.records.append(record)

    async def get(self, run_id: UUID) -> SanitizedAnalysisRecord | None:
        return next((record for record in self.records if record.run_id == run_id), None)


class MemoryUnitOfWork:
    def __init__(self) -> None:
        self.runs = MemoryRuns()
        self.events = MemoryEvents()
        self.analyses = MemoryAnalyses()
        self.committed = False

    async def __aenter__(self) -> MemoryUnitOfWork:
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        return None


class MemoryUnitOfWorkFactory:
    def __init__(self) -> None:
        self.instances: list[MemoryUnitOfWork] = []

    def __call__(self) -> MemoryUnitOfWork:
        instance = MemoryUnitOfWork()
        self.instances.append(instance)
        return instance


def _workflow(
    outputs: Mapping[str, object],
    factory: MemoryUnitOfWorkFactory,
    *,
    speech_client: FakeSpeechClient | None = None,
    workflow_settings: WorkflowSettings | None = None,
) -> AnalysisWorkflow:
    return AnalysisWorkflow(
        speech_client=speech_client or FakeSpeechClient(),  # type: ignore[arg-type]
        agents=FakeAgents(outputs),  # type: ignore[arg-type]
        policies=PolicyRegistry(),
        runtime_settings=_runtime_settings(),
        workflow_settings=workflow_settings or WorkflowSettings(code_sha="abcdef0"),
        uow_factory=cast(UnitOfWorkFactory, factory),
        now=lambda: NOW,
        monotonic=lambda: 1.0,
    )


def test_workflow_aggregates_only_deterministic_score_and_compliance_and_persists_sanitized_result() -> None:
    async def scenario() -> None:
        factory = MemoryUnitOfWorkFactory()
        response = await _workflow(_outputs(), factory).analyze(
            FileAnalyzeInput(filename="call.wav", content_type="audio/wav", content=b"RIFF"),
            request_id=RUN_ID,
        )

        assert response.quality_score.total == 80.0
        assert response.compliance.passed is False
        assert response.transcript[0].text == "Клиент спрашивает про кредит."
        assert _RAW_TRANSCRIPT_SECRET not in response.model_dump_json()
        assert len(factory.instances) == 1
        persisted = factory.instances[0]
        assert persisted.committed is True
        assert persisted.runs.statuses == [(RunStatus.PROCESSING, None), (RunStatus.COMPLETED, None)]
        assert persisted.runs.envelope is not None
        bindings = persisted.runs.envelope.provider.model_bindings
        assert {binding.agent_id: binding.reasoning_effort for binding in bindings} == {
            "classifier": "high",
            "quality": "medium",
            "compliance": None,
            "summarizer": "ultra",
        }
        assert persisted.analyses.records[0].quality_total == 80.0
        assert _RAW_TRANSCRIPT_SECRET not in "\n".join(event.model_dump_json() for event in persisted.events.events)
        assert [event.sequence for event in persisted.events.events] == list(range(1, len(persisted.events.events) + 1))
        assert all(
            current.previous_hash == previous.current_hash
            for previous, current in zip(persisted.events.events, persisted.events.events[1:])
        )

    asyncio.run(scenario())


def test_workflow_returns_structured_agent_failure_without_partial_analysis() -> None:
    async def scenario() -> None:
        factory = MemoryUnitOfWorkFactory()
        outputs = _outputs()
        outputs["compliance"] = AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)
        with pytest.raises(DomainError) as error:
            await _workflow(outputs, factory).analyze(
                FileAnalyzeInput(filename="call.wav", content_type="audio/wav", content=b"RIFF"),
                request_id=RUN_ID,
            )

        assert error.value.code is ErrorCode.AGENT_FAILURE
        assert len(factory.instances) == 1
        persisted = factory.instances[0]
        assert persisted.analyses.records == []
        assert persisted.runs.statuses == [
            (RunStatus.PROCESSING, None),
            (RunStatus.FAILED, ErrorCode.AGENT_FAILURE),
        ]
        assert persisted.events.events[-1].event_type.value == "RunFailed"

    asyncio.run(scenario())


def test_workflow_enforces_global_deadline_during_speech_before_any_agent_or_persistence() -> None:
    async def scenario() -> None:
        factory = MemoryUnitOfWorkFactory()
        workflow = _workflow(
            _outputs(),
            factory,
            speech_client=FakeSpeechClient(delay_seconds=0.05),
            workflow_settings=WorkflowSettings(code_sha="abcdef0", deadline_seconds=0.001),
        )
        with pytest.raises(DomainError) as error:
            await workflow.analyze(
                FileAnalyzeInput(filename="call.wav", content_type="audio/wav", content=b"RIFF"),
                request_id=RUN_ID,
            )

        assert error.value.code is ErrorCode.DEADLINE_EXCEEDED
        assert factory.instances == []

    asyncio.run(scenario())
