"""Shared use case: speech → four isolated agents → deterministic aggregation → persistence."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Protocol
from uuid import UUID, uuid4

from mtbank_ai.agent_runtime import AgentFailureCode, AgentRuntimeError, EventSink, LifecycleRecorder
from mtbank_ai.agents import CoreAgents
from mtbank_ai.agents.tools import AgentId
from mtbank_ai.application.ports import AnalyzeInput, FileAnalyzeInput, UrlAnalyzeInput
from mtbank_ai.config import AgentRuntimeSettings, WorkflowSettings
from mtbank_ai.domain.agents import ClassificationResult, ComplianceAssessment, QualityAssessment, SummaryResult
from mtbank_ai.domain.analysis import AnalysisVersions, AnalyzeResponse, SanitizedAnalysisRecord
from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.domain.events import LifecycleEventType, RunEvent, RunStatus
from mtbank_ai.domain.transcript import TranscriptSnapshot
from mtbank_ai.evidence.envelope import (
    MediaDescriptor,
    ModelBinding,
    PrivacyPolicy,
    ProviderFingerprint,
    RevisionSet,
    RunBudget,
    RunEnvelope,
    RunSource,
)
from mtbank_ai.observability import Telemetry
from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.speech.client import SpeechTranscriptionPort
from mtbank_ai.speech.contracts import SpeechFile
from mtbank_ai.storage.repositories import AsyncUnitOfWork
from mtbank_ai.workflow.aggregation import AggregationError, aggregate_analysis
from mtbank_ai.workflow.fetch import FetchedUrlMedia, SafeUrlFetcher, UrlFetchError, UrlFetchFailure, UrlFetchPolicy


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> AsyncUnitOfWork: ...


@dataclass(frozen=True, slots=True)
class _AgentOutputs:
    classifier: ClassificationResult
    quality: QualityAssessment
    compliance: ComplianceAssessment
    summarizer: SummaryResult


class _PersistenceFailure(RuntimeError):
    """Transaction boundary недоступен; детали DB не выходят за workflow."""


class _EventCollector(EventSink):
    """Сериализует параллельные agent events в единую hash chain одного run."""

    def __init__(self, run_id: UUID, *, now: Callable[[], datetime]) -> None:
        self._events: list[RunEvent] = []
        self._lock = asyncio.Lock()
        collector = self

        class _Sink:
            async def append(self, event: RunEvent) -> None:
                del self
                collector._events.append(event)

        self._recorder = LifecycleRecorder(run_id=run_id, sink=_Sink(), now=now)

    @property
    def events(self) -> tuple[RunEvent, ...]:
        return tuple(self._events)

    async def record(
        self,
        event_type: LifecycleEventType,
        *,
        component: str = "workflow",
        payload: dict[str, str | int | float | bool | None] | None = None,
    ) -> RunEvent:
        async with self._lock:
            return await self._recorder.record(event_type, component=component, payload=payload)

    async def append(self, event: RunEvent) -> None:
        """Принимает sanitized event из agent runtime и перенумеровывает globally."""

        payload = {item.key: item.value for item in event.payload.fields}
        await self.record(event.event_type, component="core_agent", payload=payload)


class AnalysisWorkflow:
    """Production use case, одинаковый для REST и OpenWebUI Pipeline."""

    def __init__(
        self,
        *,
        speech_client: SpeechTranscriptionPort,
        agents: CoreAgents,
        policies: PolicyRegistry,
        runtime_settings: AgentRuntimeSettings,
        workflow_settings: WorkflowSettings,
        uow_factory: UnitOfWorkFactory,
        url_fetcher: SafeUrlFetcher | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._speech_client = speech_client
        self._agents = agents
        self._policies = policies
        self._runtime_settings = runtime_settings
        self._workflow_settings = workflow_settings
        self._uow_factory = uow_factory
        self._url_fetcher = url_fetcher or SafeUrlFetcher(
            UrlFetchPolicy(
                max_bytes=workflow_settings.max_url_bytes,
                timeout_seconds=workflow_settings.url_timeout_seconds,
                max_redirects=workflow_settings.url_max_redirects,
            )
        )
        self._now = now
        self._monotonic = monotonic
        self._telemetry = telemetry or Telemetry()

    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        return await self._analyze(source, request_id=request_id, source_override=None)

    async def analyze_openwebui(self, source: FileAnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        """Pipeline entry point, отличающий trusted OpenWebUI input в envelope."""

        return await self._analyze(source, request_id=request_id, source_override=RunSource.OPENWEBUI)

    async def _analyze(
        self,
        source: AnalyzeInput,
        *,
        request_id: UUID,
        source_override: RunSource | None,
    ) -> AnalyzeResponse:
        created_at = self._now()
        run_id = uuid4()
        event_collector = _EventCollector(run_id, now=self._now)
        started = self._monotonic()
        deadline_at = created_at + timedelta(seconds=self._workflow_settings.deadline_seconds)
        await event_collector.record(LifecycleEventType.RUN_STARTED, payload={"request_id": str(request_id)})
        await event_collector.record(LifecycleEventType.SPEECH_STARTED)
        transcript, source_kind, media = await self._transcribe(
            source,
            event_collector,
            deadline_at=deadline_at,
        )
        if source_override is not None:
            source_kind = source_override
        await event_collector.record(
            LifecycleEventType.SPEECH_COMPLETED,
            payload={"transcript_id": str(transcript.transcript_id), "audio_sha256": transcript.audio_sha256},
        )

        prompt_bundle_hash = self._agents.prompt_bundle_hash(transcript)
        envelope = self._build_envelope(
            run_id=run_id,
            request_id=request_id,
            source=source_kind,
            transcript=transcript,
            media=media,
            created_at=created_at,
            deadline_at=deadline_at,
            prompt_bundle_hash=prompt_bundle_hash,
        )
        try:
            outputs = await self._run_agents(
                transcript,
                run_id=run_id,
                created_at=created_at,
                deadline_at=deadline_at,
                event_sink=event_collector,
            )
            processing_ms = max(0, int((self._monotonic() - started) * 1_000))
            aggregated = aggregate_analysis(
                transcript,
                classification=outputs.classifier,
                quality=outputs.quality,
                compliance=outputs.compliance,
                summary=outputs.summarizer,
                policies=self._policies,
                run_id=run_id,
                versions=AnalysisVersions(
                    code_sha=self._workflow_settings.code_sha,
                    prompt_bundle_hash=prompt_bundle_hash,
                    taxonomy_version=f"taxonomy/{self._policies.taxonomy.version}",
                    quality_rubric_version=f"quality/{self._policies.quality.version}",
                    compliance_policy_version=f"compliance/{self._policies.compliance.version}",
                    asr=transcript.asr_metadata.asr,
                    alignment=transcript.asr_metadata.alignment,
                    diarization=transcript.asr_metadata.diarization,
                ),
                processing_ms=processing_ms,
            )
            await event_collector.record(LifecycleEventType.AGGREGATION_COMPLETED)
            await event_collector.record(LifecycleEventType.RUN_COMPLETED)
            self._telemetry.metrics.gauge(
                "mtbank_quality_total",
                aggregated.response.quality_score.total,
                topic=aggregated.response.classification.topic,
            )
            self._telemetry.metrics.increment(
                "mtbank_compliance_calls_total", passed=aggregated.response.compliance.passed
            )
            self._telemetry.metrics.increment(
                "mtbank_topic_calls_total", topic=aggregated.response.classification.topic
            )
            with self._telemetry.span("persistence.save"):
                await self._persist(
                    envelope,
                    event_collector.events,
                    status=RunStatus.COMPLETED,
                    record=aggregated.sanitized_record,
                    error_code=None,
                )
            return aggregated.response
        except _PersistenceFailure:
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from None
        except DomainError as error:
            await event_collector.record(LifecycleEventType.RUN_FAILED, payload={"failure_code": error.code.value})
            await self._persist_failure(envelope, event_collector.events, error.code)
            raise
        except AggregationError:
            error = DomainError(ErrorCode.AGENT_FAILURE)
            await event_collector.record(LifecycleEventType.RUN_FAILED, payload={"failure_code": error.code.value})
            await self._persist_failure(envelope, event_collector.events, error.code)
            raise error from None
        except Exception as error:
            mapped = _map_agent_failure(error)
            await event_collector.record(LifecycleEventType.RUN_FAILED, payload={"failure_code": mapped.code.value})
            await self._persist_failure(envelope, event_collector.events, mapped.code)
            raise mapped from None

    async def close(self) -> None:
        close = getattr(self._agents, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _transcribe(
        self,
        source: AnalyzeInput,
        event_collector: _EventCollector,
        *,
        deadline_at: datetime,
    ) -> tuple[TranscriptSnapshot, RunSource, FetchedUrlMedia | FileAnalyzeInput]:
        try:
            remaining_seconds = (deadline_at - self._now()).total_seconds()
            if remaining_seconds <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining_seconds):
                if isinstance(source, FileAnalyzeInput):
                    media: FetchedUrlMedia | FileAnalyzeInput = source
                    source_kind = RunSource.REST_FILE
                elif isinstance(source, UrlAnalyzeInput):
                    with self._telemetry.span("url.fetch"):
                        media = await self._url_fetcher.fetch(str(source.url))
                    source_kind = RunSource.REST_URL
                else:
                    raise DomainError(ErrorCode.INVALID_INPUT)
                with self._telemetry.span("speech.transcribe"):
                    response = await self._speech_client.transcribe(
                        SpeechFile(filename=media.filename, content_type=media.content_type, content=media.content)
                    )
                return response.transcript, source_kind, media
        except DomainError:
            self._telemetry.metrics.increment("mtbank_speech_errors_total", reason="domain_error")
            await event_collector.record(LifecycleEventType.SPEECH_FAILED, payload={"failure_code": "domain_error"})
            raise
        except UrlFetchError as error:
            await event_collector.record(
                LifecycleEventType.SPEECH_FAILED,
                payload={"failure_code": error.failure.value},
            )
            raise _map_fetch_failure(error) from None
        except TimeoutError:
            await event_collector.record(LifecycleEventType.SPEECH_FAILED, payload={"failure_code": "timeout"})
            raise DomainError(ErrorCode.DEADLINE_EXCEEDED) from None
        except Exception:
            await event_collector.record(LifecycleEventType.SPEECH_FAILED, payload={"failure_code": "unavailable"})
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from None

    async def _run_agents(
        self,
        transcript: TranscriptSnapshot,
        *,
        run_id: UUID,
        created_at: datetime,
        deadline_at: datetime,
        event_sink: EventSink,
    ) -> _AgentOutputs:
        executions: dict[str, asyncio.Task] = {}

        async def run_agent(agent_id: AgentId):  # type: ignore[no-untyped-def]
            with self._telemetry.span("agent.run", agent_id=agent_id):
                return await self._agents.runner(agent_id).run(
                    transcript,
                    run_id=run_id,
                    run_version="analysis/v1",
                    created_at=created_at,
                    deadline_at=deadline_at,
                    event_sink=event_sink,
                )

        try:
            async with asyncio.TaskGroup() as group:
                for agent_id in self._agents.agent_ids:
                    executions[agent_id] = group.create_task(run_agent(agent_id))
        except BaseException as error:
            raise _first_taskgroup_error(error) from None
        outputs = {agent_id: task.result().output for agent_id, task in executions.items()}
        classifier = outputs.get("classifier")
        quality = outputs.get("quality")
        compliance = outputs.get("compliance")
        summary = outputs.get("summarizer")
        if not isinstance(classifier, ClassificationResult):
            raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)
        if not isinstance(quality, QualityAssessment):
            raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)
        if not isinstance(compliance, ComplianceAssessment):
            raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)
        if not isinstance(summary, SummaryResult):
            raise AgentRuntimeError(AgentFailureCode.TERMINAL_SUBMIT_INVALID)
        return _AgentOutputs(
            classifier=classifier,
            quality=quality,
            compliance=compliance,
            summarizer=summary,
        )

    def _build_envelope(
        self,
        *,
        run_id: UUID,
        request_id: UUID,
        source: RunSource,
        transcript: TranscriptSnapshot,
        media: FetchedUrlMedia | FileAnalyzeInput,
        created_at: datetime,
        deadline_at: datetime,
        prompt_bundle_hash: str,
    ) -> RunEnvelope:
        bindings = tuple(
            ModelBinding(
                agent_id=agent_id,
                provider_id=self._workflow_settings.provider_id,
                model_id=configuration.model_id,
                reasoning_effort=configuration.reasoning_effort,
            )
            for agent_id, configuration in self._agents.model_configurations().items()
        )
        return RunEnvelope(
            run_id=run_id,
            request_id=request_id,
            correlation_id=request_id,
            source=source,
            input_media=MediaDescriptor(
                sha256=transcript.audio_sha256,
                mime_type=media.content_type,
                duration_seconds=transcript.duration_seconds,
                sample_rate_hz=self._workflow_settings.normalized_sample_rate_hz,
                channels=self._workflow_settings.normalized_channels,
            ),
            provider=ProviderFingerprint(model_bindings=bindings),
            revisions=RevisionSet(
                code_sha=self._workflow_settings.code_sha,
                prompt_bundle_hash=prompt_bundle_hash,
                taxonomy_version=f"taxonomy/{self._policies.taxonomy.version}",
                quality_rubric_version=f"quality/{self._policies.quality.version}",
                compliance_policy_version=f"compliance/{self._policies.compliance.version}",
                asr=transcript.asr_metadata.asr,
                alignment=transcript.asr_metadata.alignment,
                diarization=transcript.asr_metadata.diarization,
            ),
            budget=RunBudget(
                deadline_at=deadline_at,
                max_llm_turns=self._runtime_settings.default_max_turns * len(self._agents.agent_ids),
                max_total_tokens=(
                    self._runtime_settings.default_max_input_tokens + self._runtime_settings.default_max_output_tokens
                )
                * len(self._agents.agent_ids),
                max_cost_usd=self._runtime_settings.default_max_cost_usd * Decimal(len(self._agents.agent_ids)),
            ),
            privacy=PrivacyPolicy(
                mode=self._workflow_settings.privacy_mode,
                raw_audio_retention_seconds=self._workflow_settings.raw_audio_retention_seconds,
                evidence_retention_days=self._workflow_settings.evidence_retention_days,
                allow_full_content_evidence=False,
            ),
            created_at=created_at,
        )

    async def _persist(
        self,
        envelope: RunEnvelope,
        events: tuple[RunEvent, ...],
        *,
        status: RunStatus,
        record: SanitizedAnalysisRecord | None,
        error_code: ErrorCode | None,
    ) -> None:
        try:
            async with self._uow_factory() as uow:
                await uow.runs.create(envelope)
                await uow.runs.set_status(envelope.run_id, RunStatus.PROCESSING)
                for event in events:
                    await uow.events.append(event)
                if record is not None:
                    await uow.analyses.save_sanitized(record)
                await uow.runs.set_status(envelope.run_id, status, error_code=error_code)
                await uow.commit()
        except Exception:
            raise _PersistenceFailure from None

    async def _persist_failure(
        self,
        envelope: RunEnvelope,
        events: tuple[RunEvent, ...],
        error_code: ErrorCode,
    ) -> None:
        try:
            await self._persist(
                envelope,
                events,
                status=RunStatus.FAILED,
                record=None,
                error_code=error_code,
            )
        except _PersistenceFailure:
            return


AnalyzeCallUseCase = AnalysisWorkflow


def _map_fetch_failure(error: UrlFetchError) -> DomainError:
    if error.failure is UrlFetchFailure.PAYLOAD_TOO_LARGE:
        return DomainError(ErrorCode.PAYLOAD_TOO_LARGE)
    if error.failure is UrlFetchFailure.UNSUPPORTED_MEDIA:
        return DomainError(ErrorCode.UNSUPPORTED_MEDIA)
    if error.failure is UrlFetchFailure.TIMEOUT:
        return DomainError(ErrorCode.DEADLINE_EXCEEDED)
    if error.failure is UrlFetchFailure.UNAVAILABLE:
        return DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    return DomainError(ErrorCode.INVALID_URL)


def _first_taskgroup_error(error: BaseException) -> BaseException:
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            found = _first_taskgroup_error(nested)
            if found is not error:
                return found
    return error


def _map_agent_failure(error: Exception) -> DomainError:
    if not isinstance(error, AgentRuntimeError):
        return DomainError(ErrorCode.AGENT_FAILURE)
    if error.code is AgentFailureCode.PROVIDER_RATE_LIMITED:
        return DomainError(ErrorCode.QUOTA_EXCEEDED)
    if error.code in {AgentFailureCode.DEADLINE_EXCEEDED, AgentFailureCode.PROVIDER_TIMEOUT}:
        return DomainError(ErrorCode.DEADLINE_EXCEEDED)
    if error.code is AgentFailureCode.CIRCUIT_OPEN:
        return DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    if error.code in {
        AgentFailureCode.PROVIDER_AUTHENTICATION,
        AgentFailureCode.PROVIDER_PERMISSION,
        AgentFailureCode.PROVIDER_INVALID_REQUEST,
        AgentFailureCode.PROVIDER_TRANSPORT,
        AgentFailureCode.PROVIDER_SERVER,
    }:
        return DomainError(ErrorCode.PROVIDER_FAILURE)
    return DomainError(ErrorCode.AGENT_FAILURE)
