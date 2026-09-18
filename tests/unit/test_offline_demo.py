from __future__ import annotations

import asyncio
import json

from scripts.demo_offline import load_transcript, run_demo


def test_offline_demo_produces_grounded_analysis_without_a_model() -> None:
    async def scenario() -> None:
        response, stages = await run_demo()
        transcript = load_transcript()
        segment_ids = {str(segment.id) for segment in transcript.segments}

        assert response.classification.topic == "кредиты"
        assert response.quality_score.total == 100.0
        assert response.compliance.passed is True
        assert response.compliance.issues == ()
        assert response.quality_score.checklist.greeting is True
        assert len(stages) == 4

        evidence = {
            str(identifier)
            for item in (
                response.quality_score.details.greeting,
                response.quality_score.details.need_detection,
                response.quality_score.details.solution_provided,
                response.quality_score.details.farewell,
            )
            for identifier in item.evidence_segment_ids
        } | {str(identifier) for identifier in response.classification.evidence_segment_ids}
        assert evidence <= segment_ids
        assert {str(segment.id) for segment in response.transcript} == segment_ids

    asyncio.run(scenario())


def test_offline_demo_is_deterministic() -> None:
    async def scenario() -> None:
        first, _ = await run_demo()
        second, _ = await run_demo()

        assert json.loads(first.model_dump_json()) == json.loads(second.model_dump_json())

    asyncio.run(scenario())


def test_offline_demo_exposes_the_fail_closed_compliance_path() -> None:
    async def scenario() -> None:
        response, stages = await run_demo(inject_violation=True)

        assert response.compliance.passed is False
        blocking = [issue for issue in response.compliance.issues if issue.severity.value == "blocking"]
        assert [issue.rule_id for issue in response.compliance.issues] == ["no_unconditional_guarantee"]
        assert blocking
        injected_ids = {str(issue.evidence_segment_ids[0]) for issue in blocking}
        assert injected_ids <= {str(segment.id) for segment in response.transcript}
        assert any("injected" in stage for stage in stages)

    asyncio.run(scenario())
