#!/usr/bin/env python3
"""Deterministic offline walkthrough: transcript contract -> four agents -> deterministic fan-in.

OFFLINE DEMO. No model call, no GPU, no network, no credentials.

Executed for real:
  * the `TranscriptSnapshot` contract and evidence grounding against immutable segment IDs;
  * reviewed prompts, the tool registry (`transcript_read` plus one policy tool per agent)
    and typed terminal validation of every agent result;
  * deterministic quality/compliance fan-in and `AnalyzeResponse` construction.

Replaced:
  * the OpenAI-compatible gateway. `ScriptedGateway` answers with fixed tool calls whose
    arguments are derived from the reference transcript, so the walkthrough exercises the
    real contracts without a provider.

This is not a model-quality evaluation. Measured ASR and agent runs live in
`release-evidence/final-115/`.

Usage:
    uv run python scripts/demo_offline.py
    uv run python scripts/demo_offline.py --json-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import SecretStr

from mtbank_ai.agent_runtime import ModelRequest, ModelResponse, ModelToolCall, ModelUsage
from mtbank_ai.agents import CoreAgents
from mtbank_ai.config import AgentRuntimeSettings, GatewayModelSettings, GatewaySettings
from mtbank_ai.domain.agents import (
    ClassificationResult,
    ComplianceAssessment,
    QualityAssessment,
    SummaryResult,
)
from mtbank_ai.domain.analysis import AnalysisVersions, AnalyzeResponse
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import (
    ASRMetadata,
    RoleAssignment,
    RoleResolution,
    RoleResolutionSource,
    SpeakerRole,
    TranscriptSegment,
    TranscriptSnapshot,
)
from mtbank_ai.policies import PolicyRegistry
from mtbank_ai.workflow.aggregation import aggregate_analysis

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "test_data" / "references" / "synthetic-credit-consultation.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "demo-offline-response.json"
RUN_ID = uuid5(NAMESPACE_URL, "offline-demo:run")
NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
REVISION = ComponentRevision(
    package="offline-demo",
    package_version="0.0.0",
    model_id="offline-demo",
    model_revision="offline-demo/v1",
)
GREETING_MARKERS = ("добрый день", "добрый вечер", "здравствуйте")
FAREWELL_MARKERS = ("всего доброго", "до свидания", "хорошего дня", "всего хорошего", "обращайтесь")
SOLUTION_MARKERS = ("условия", "условиях", "ставк", "оформ", "заявк", "решени", "перевод", "блокир")
GUARANTEE_MARKERS = ("гарантируем одобрение", "гарантирую одобрение", "гарантированное одобрение")
SECRET_REQUEST_MARKERS = ("назовите код", "сообщите код", "продиктуйте", "скажите пароль", "cvv", "cvc", "пин-код")
NEGATION_MARKERS = (
    "не сообщайте",
    "не называйте",
    "не диктуйте",
    "не запрашива",
    "не пересылайте",
    "не повторяйте",
    "никогда не",
    "без гарантии",
    "нельзя гарантировать",
)
INJECTED_VIOLATION = "Мы гарантируем одобрение кредита по этой ставке, можете не сомневаться."


def _segment_id(raw_id: object) -> Any:
    return uuid5(NAMESPACE_URL, f"offline-demo:{raw_id}")


def load_transcript(path: Path = REFERENCE) -> TranscriptSnapshot:
    """Build the production transcript contract from the synthetic reference corpus."""

    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"reference transcript is unavailable: {path}") from error
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise SystemExit(f"reference transcript has no segments: {path}")

    segments: list[TranscriptSegment] = []
    evidence_by_speaker: dict[str, list[Any]] = {}
    role_by_speaker: dict[str, SpeakerRole] = {}
    for raw in raw_segments:
        identifier = _segment_id(raw["id"])
        role = SpeakerRole(raw["speaker"])
        text = str(raw["text"]).strip()
        speaker_id = f"reference-{role.name.lower()}"
        evidence_by_speaker.setdefault(speaker_id, []).append(identifier)
        role_by_speaker[speaker_id] = role
        segments.append(
            TranscriptSegment(
                id=identifier,
                original_speaker_id=speaker_id,
                speaker=role,
                role_confidence=1.0,
                start=float(raw["start"]),
                end=float(raw["end"]),
                # Synthetic corpus: there is nothing to redact, so both fields carry the same text.
                text=text,
                redacted_text=text,
            )
        )

    assignments = [
        RoleAssignment(
            original_speaker_id=speaker_id,
            role=role_by_speaker[speaker_id],
            confidence=1.0,
            evidence_segment_ids=tuple(evidence_by_speaker[speaker_id]),
            source=RoleResolutionSource.METADATA,
            resolution_evidence="reference corpus authors the speaker labels",
        )
        for speaker_id in sorted(evidence_by_speaker)
    ]
    return TranscriptSnapshot(
        transcript_id=uuid5(NAMESPACE_URL, "offline-demo:transcript"),
        audio_sha256="0" * 64,
        revision="transcript/offline-demo",
        language="ru",
        duration_seconds=max(segment.end for segment in segments),
        segments=tuple(segments),
        role_resolution=RoleResolution(assignments=tuple(assignments), needs_review=False),
        asr_metadata=ASRMetadata(
            asr=REVISION,
            alignment=REVISION,
            diarization=REVISION,
            language="ru",
            processing_ms=0,
        ),
        created_at=NOW,
    )


def with_injected_violation(transcript: TranscriptSnapshot) -> TranscriptSnapshot:
    """Append one local fixture line that violates a blocking compliance rule.

    It demonstrates the fail-closed path (a blocking issue must not be reported as a pass).
    The line is not part of the reference corpus and is labelled as injected in the output.
    """

    extra = TranscriptSegment(
        id=uuid5(NAMESPACE_URL, "offline-demo:injected-violation"),
        original_speaker_id="reference-operator",
        speaker=SpeakerRole.OPERATOR,
        role_confidence=1.0,
        start=transcript.duration_seconds,
        end=transcript.duration_seconds + 4.0,
        text=INJECTED_VIOLATION,
        redacted_text=INJECTED_VIOLATION,
    )
    assignments = tuple(
        assignment.model_copy(update={"evidence_segment_ids": (*assignment.evidence_segment_ids, extra.id)})
        if assignment.original_speaker_id == extra.original_speaker_id
        else assignment
        for assignment in transcript.role_resolution.assignments
    )
    return transcript.model_copy(
        update={
            "segments": (*transcript.segments, extra),
            "duration_seconds": extra.end,
            "role_resolution": RoleResolution(
                assignments=assignments,
                needs_review=transcript.role_resolution.needs_review,
                agent_provenance=transcript.role_resolution.agent_provenance,
            ),
        }
    )


def runtime_settings() -> AgentRuntimeSettings:
    """Gateway settings for the scripted provider.

    The key below satisfies the configuration validator (length, charset, no placeholder
    markers). It is a local non-credential: the scripted provider never opens a socket.
    """

    return AgentRuntimeSettings(
        gateway=GatewaySettings(
            base_url="https://offline-demo.invalid/v1",
            api_key=SecretStr("offline-demo-only-4Kq9Zm2Rt7Xw1Bv6"),
            models=GatewayModelSettings(
                default_model="classifier-model",
                classifier_model="classifier-model",
                quality_model="quality-model",
                compliance_model="compliance-model",
                summarizer_model="summarizer-model",
                input_token_cost_usd=Decimal("0"),
                output_token_cost_usd=Decimal("0"),
            ),
        )
    )


def _tool_call(name: str, *, call_id: str, arguments: dict[str, object]) -> ModelToolCall:
    return ModelToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments, ensure_ascii=False))


def _matching(segments: list[TranscriptSegment], markers: tuple[str, ...]) -> TranscriptSegment | None:
    for segment in segments:
        lowered = segment.redacted_text.lower()
        if any(marker in lowered for marker in markers):
            return segment
    return None


def _violating(segments: list[TranscriptSegment], markers: tuple[str, ...]) -> TranscriptSegment | None:
    """Match a risky phrase, ignoring coaching such as "не называйте код"."""

    for segment in segments:
        lowered = segment.redacted_text.lower()
        if any(marker in lowered for marker in markers) and not any(
            negation in lowered for negation in NEGATION_MARKERS
        ):
            return segment
    return None


def _criterion(
    segment: TranscriptSegment | None,
    fallback: TranscriptSegment,
    *,
    found: float,
    missing: float,
) -> dict[str, object]:
    """Assess one rubric criterion; a failed criterion still cites the segment that shows the gap."""

    if segment is not None:
        return {
            "passed": True,
            "confidence": found,
            "evidence_segment_ids": [str(segment.id)],
            "rationale": "Подтверждено репликой reference-транскрипта.",
        }
    return {
        "passed": False,
        "confidence": missing,
        "evidence_segment_ids": [str(fallback.id)],
        "rationale": "В reference-транскрипте нет подтверждающей реплики.",
    }


@dataclass(slots=True)
class ScriptedGateway:
    """Deterministic stand-in for the OpenAI-compatible gateway.

    Turn 1 asks for the transcript plus one reviewed policy document, turn 2 submits the
    typed terminal output. Arguments are derived from the transcript, never invented.
    """

    operator: list[TranscriptSegment]
    client: list[TranscriptSegment]
    turns: dict[str, int] = field(default_factory=dict)

    def _issues(self) -> list[dict[str, object]]:
        issues: list[dict[str, object]] = []
        guarantee = _violating(self.operator, GUARANTEE_MARKERS)
        if guarantee is not None:
            issues.append(
                {
                    "rule_id": "no_unconditional_guarantee",
                    "severity": "blocking",
                    "evidence_segment_ids": [str(guarantee.id)],
                    "explanation": "Оператор использует формулировку с безусловной гарантией.",
                }
            )
        sensitive = _violating(self.operator, SECRET_REQUEST_MARKERS)
        if sensitive is not None:
            issues.append(
                {
                    "rule_id": "no_sensitive_data_request",
                    "severity": "blocking",
                    "evidence_segment_ids": [str(sensitive.id)],
                    "explanation": "В диалоге упоминается запрос чувствительных данных.",
                }
            )
        return issues

    def _summary(self) -> dict[str, object]:
        unique: list[TranscriptSegment] = []
        for segment in self.operator[:1] + self.client[:1] + self.operator[-1:]:
            if segment not in unique:
                unique.append(segment)
        email = _matching(self.client, ("@", "почт", "email"))
        return {
            "sentences": [segment.redacted_text for segment in unique],
            "fact_segment_ids": [str(segment.id) for segment in unique],
            "action_items": (
                [
                    {
                        "text": "Отправить клиенту материалы на указанный в разговоре адрес.",
                        "evidence_segment_ids": [str(email.id)],
                    }
                ]
                if email is not None
                else []
            ),
        }

    def _terminal_call(self, model_id: str) -> ModelToolCall:
        if model_id == "classifier-model":
            credit = [segment for segment in self.operator + self.client if "кредит" in segment.redacted_text.lower()]
            arguments: dict[str, object] = {
                "topic": "кредиты" if credit else "другое",
                "priority": "medium" if credit else "low",
                "confidence": 0.9 if credit else 0.5,
                "evidence_segment_ids": [str(segment.id) for segment in credit[:3]],
                "rationale": "Тема определена по упоминаниям кредитных продуктов в репликах.",
            }
            name = "submit_classification"
        elif model_id == "quality-model":
            arguments = {
                "greeting": _criterion(
                    _matching(self.operator, GREETING_MARKERS), self.operator[0], found=0.95, missing=0.4
                ),
                "need_detection": _criterion(
                    next((segment for segment in self.client if "?" in segment.redacted_text), None),
                    self.client[0],
                    found=0.9,
                    missing=0.4,
                ),
                "solution_provided": _criterion(
                    _matching(self.operator, SOLUTION_MARKERS), self.operator[-1], found=0.85, missing=0.35
                ),
                "farewell": _criterion(
                    _matching(list(reversed(self.operator)), FAREWELL_MARKERS),
                    self.operator[-1],
                    found=0.8,
                    missing=0.3,
                ),
            }
            name = "submit_quality"
        elif model_id == "compliance-model":
            arguments = {"issues": self._issues()}
            name = "submit_compliance"
        else:
            arguments = self._summary()
            name = "submit_summary"
        return _tool_call(name, call_id=f"{model_id}-submit", arguments=arguments)

    @staticmethod
    def _retrieval_calls(model_id: str) -> tuple[ModelToolCall, ModelToolCall]:
        policy_tool = {
            "classifier-model": "taxonomy_get",
            "quality-model": "quality_rubric_get",
            "compliance-model": "compliance_rules_list",
            "summarizer-model": "transcript_statistics",
        }[model_id]
        return (
            _tool_call("transcript_read", call_id=f"{model_id}-read", arguments={}),
            _tool_call(policy_tool, call_id=f"{model_id}-policy", arguments={}),
        )

    async def complete(self, request: ModelRequest, *, deadline_at: datetime) -> ModelResponse:
        del deadline_at
        turn = self.turns.get(request.model_id, 0)
        self.turns[request.model_id] = turn + 1
        tool_calls = self._retrieval_calls(request.model_id) if turn == 0 else (self._terminal_call(request.model_id),)
        return ModelResponse(
            request_id=None,
            model_id=request.model_id,
            finish_reason="tool_calls",
            tool_calls=tool_calls,
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            latency_ms=0,
            has_text_content=False,
        )


async def run_demo(*, reference: Path = REFERENCE, inject_violation: bool = False) -> tuple[AnalyzeResponse, list[str]]:
    """Run the four real agents against the scripted gateway and the real fan-in."""

    transcript = load_transcript(reference)
    if inject_violation:
        transcript = with_injected_violation(transcript)
    policies = PolicyRegistry()
    gateway = ScriptedGateway(
        operator=[segment for segment in transcript.segments if segment.speaker is SpeakerRole.OPERATOR],
        client=[segment for segment in transcript.segments if segment.speaker is SpeakerRole.CLIENT],
    )
    agents = CoreAgents(model_client=gateway, runtime_settings=runtime_settings(), policies=policies)
    started_at = datetime.now(UTC)
    results = await asyncio.gather(
        *(
            agents.runner(agent_id).run(
                transcript,
                run_id=RUN_ID,
                run_version="analysis/offline-demo",
                created_at=started_at,
                deadline_at=started_at + timedelta(seconds=30),
            )
            for agent_id in agents.agent_ids
        )
    )
    outputs = {result.agent_id: result.output for result in results}
    classification, quality = outputs["classifier"], outputs["quality"]
    compliance, summarizer = outputs["compliance"], outputs["summarizer"]
    if not (
        isinstance(classification, ClassificationResult)
        and isinstance(quality, QualityAssessment)
        and isinstance(compliance, ComplianceAssessment)
        and isinstance(summarizer, SummaryResult)
    ):
        raise SystemExit("agent terminal output does not match the expected contract")
    aggregated = aggregate_analysis(
        transcript,
        classification=classification,
        quality=quality,
        compliance=compliance,
        summary=summarizer,
        policies=policies,
        run_id=RUN_ID,
        versions=AnalysisVersions(
            code_sha="offline-demo",
            prompt_bundle_hash=agents.prompt_bundle_hash(transcript),
            taxonomy_version=f"taxonomy/{policies.taxonomy.version}",
            quality_rubric_version=f"quality/{policies.quality.version}",
            compliance_policy_version=f"compliance/{policies.compliance.version}",
            asr=transcript.asr_metadata.asr,
            alignment=transcript.asr_metadata.alignment,
            diarization=transcript.asr_metadata.diarization,
        ),
        processing_ms=0,
    )
    stages = [
        f"corpus            : {reference.name} ({len(transcript.segments)} segments, "
        f"{transcript.duration_seconds:.1f} s, synthetic"
        + (", one injected violating line)" if inject_violation else ")"),
        "speech contract   : supplied by the reference corpus (ASR itself is not part of this demo)",
        f"agents            : {', '.join(sorted(agents.agent_ids))} (parallel, "
        f"{sum(gateway.turns.values())} scripted model turns)",
        f"fan-in            : quality_total={aggregated.response.quality_score.total}, "
        f"compliance_passed={aggregated.response.compliance.passed}, "
        f"needs_review={aggregated.response.meta.needs_review}",
    ]
    return aggregated.response, stages


def _print_report(response: AnalyzeResponse, stages: list[str]) -> None:
    print("Offline demo — scripted gateway, synthetic reference corpus, no model call.")
    print("Real code paths: transcript contract, agent tools, typed terminal validation, deterministic fan-in.")
    print()
    for stage in stages:
        print(f"  {stage}")
    print()
    print(f"  topic        : {response.classification.topic} / {response.classification.priority}")
    print(f"  quality      : {response.quality_score.total} (policy {response.quality_score.policy_version})")
    details = response.quality_score.details
    for name, item in (
        ("greeting", details.greeting),
        ("need_detection", details.need_detection),
        ("solution", details.solution_provided),
        ("farewell", details.farewell),
    ):
        print(f"    - {name:<15}: passed={item.passed} evidence={len(item.evidence_segment_ids)}")
    print(f"  compliance   : passed={response.compliance.passed} issues={len(response.compliance.issues)}")
    print(f"  summary      : {response.summary}")
    print(f"  action_items : {len(response.action_items)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic offline walkthrough (no model, no GPU, no network).")
    parser.add_argument("--json-only", action="store_true", help="print only the resulting AnalyzeResponse JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to write the response JSON")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="reference transcript to replay")
    parser.add_argument(
        "--inject-violation",
        action="store_true",
        help="append one local fixture line that violates a blocking compliance rule",
    )
    arguments = parser.parse_args(argv)

    response, stages = asyncio.run(run_demo(reference=arguments.reference, inject_violation=arguments.inject_violation))
    payload = response.model_dump_json(indent=2)

    if not arguments.json_only:
        _print_report(response, stages)
    print(payload)

    written = True
    try:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    except OSError as error:
        written = False
        print(f"warning: response was not written to {arguments.output}: {error}", file=sys.stderr)
    if written and not arguments.json_only:
        print(f"response written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
