from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from email.message import Message
from typing import cast
from urllib.error import HTTPError
from uuid import UUID

import pytest

from mtbank_ai.application.ports import FileAnalyzeInput
from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.workflow.pipeline_adapter import OpenWebUIAnalysisAdapter, render_openwebui_analysis
from pipeline import ApiAnalysisClient, Pipeline, VerifiedAudio

REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")


@dataclass
class RenderableResponse:
    payload: dict[str, object]

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self.payload


class AsyncUseCase:
    def __init__(self) -> None:
        self.sources: list[FileAnalyzeInput] = []
        self.request_ids: list[UUID] = []

    async def analyze_openwebui(self, source: FileAnalyzeInput, *, request_id: UUID) -> RenderableResponse:
        self.sources.append(source)
        self.request_ids.append(request_id)
        return RenderableResponse({"summary": "готово"})


class PipelineAdapter:
    def __init__(self) -> None:
        self.sources: list[FileAnalyzeInput] = []
        self.request_ids: list[UUID] = []

    def analyze(self, source: FileAnalyzeInput, *, request_id: UUID) -> RenderableResponse:
        self.sources.append(source)
        self.request_ids.append(request_id)
        return RenderableResponse({"summary": "<script>untrusted</script>"})


class LocalAnalyzeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._offset = 0

    def __enter__(self) -> LocalAnalyzeResponse:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback

    def read(self, size: int) -> bytes:
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class LocalAnalyzeStub:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.timeouts: list[float] = []

    def __call__(self, request: object, *, timeout: float) -> LocalAnalyzeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        return LocalAnalyzeResponse(_analysis_payload())


def _analysis_payload() -> dict[str, object]:
    segment_id = "33333333-3333-4333-8333-333333333333"
    criterion = {
        "passed": True,
        "confidence": 0.9,
        "evidence_segment_ids": [segment_id],
        "rationale": "Критерий подтверждён.",
    }
    component = {
        "package": "faster-whisper",
        "package_version": "1.0.0",
        "model_id": "large-v3",
        "model_revision": "asr/v1",
    }
    return {
        "transcript": [
            {
                "id": segment_id,
                "speaker": "Оператор",
                "start": 0.0,
                "end": 1.0,
                "text": "Добрый день.",
            }
        ],
        "classification": {
            "topic": "кредиты",
            "priority": "medium",
            "confidence": 0.9,
            "evidence_segment_ids": [segment_id],
            "rationale": "Тема подтверждена.",
            "taxonomy_version": "taxonomy/v1",
        },
        "quality_score": {
            "total": 100.0,
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": True,
            },
            "details": {
                "greeting": criterion,
                "need_detection": criterion,
                "solution_provided": criterion,
                "farewell": criterion,
            },
            "policy_version": "quality/v1",
        },
        "compliance": {"passed": True, "issues": [], "policy_version": "compliance/v1"},
        "summary": "Клиент запросил информацию о кредите.",
        "action_items": ["Отправить условия."],
        "grounding": {
            "summary_evidence_segment_ids": [segment_id],
            "action_items": [{"text": "Отправить условия.", "evidence_segment_ids": [segment_id]}],
        },
        "meta": {
            "run_id": "22222222-2222-4222-8222-222222222222",
            "status": "completed",
            "versions": {
                "code_sha": "abcdef0",
                "prompt_bundle_hash": "a" * 64,
                "taxonomy_version": "taxonomy/v1",
                "quality_rubric_version": "quality/v1",
                "compliance_policy_version": "compliance/v1",
                "asr": component,
                "alignment": {**component, "package": "whisperx", "model_revision": "alignment/v1"},
                "diarization": {
                    **component,
                    "package": "pyannote.audio",
                    "model_revision": "diarization/v1",
                },
            },
            "processing_ms": 10,
            "needs_review": False,
        },
    }


def test_openwebui_adapter_bridges_sync_call_inside_running_event_loop() -> None:
    async def scenario() -> None:
        use_case = AsyncUseCase()
        adapter = OpenWebUIAnalysisAdapter(use_case)  # type: ignore[arg-type]
        source = FileAnalyzeInput(filename="call.wav", content_type="audio/wav", content=b"RIFF")

        response = cast(RenderableResponse, adapter.analyze(source, request_id=REQUEST_ID))

        assert response.payload == {"summary": "готово"}
        assert use_case.sources == [source]
        assert use_case.request_ids == [REQUEST_ID]

    asyncio.run(scenario())


def test_pipeline_adapter_uses_verified_bytes_and_escapes_public_response() -> None:
    adapter = PipelineAdapter()
    pipeline = Pipeline(analysis_adapter=adapter)  # type: ignore[arg-type]
    verified = VerifiedAudio(
        name="authoritative.wav",
        content_type="audio/wav",
        content=b"RIFF\x00\x00\x00\x00WAVE",
    )

    rendered = pipeline._render_analysis(verified)

    assert adapter.sources[0].filename == "authoritative.wav"
    assert adapter.sources[0].content == verified.content
    assert adapter.sources[0].content_type == "audio/wav"
    assert adapter.request_ids[0].version == 4
    assert "&lt;script&gt;untrusted&lt;/script&gt;" in rendered
    assert "<script>" not in rendered
    assert rendered == render_openwebui_analysis(
        RenderableResponse({"summary": "<script>untrusted</script>"}),  # type: ignore[arg-type]
        display_name=pipeline.name,
    )


def test_internal_api_adapter_sends_verified_audio_only_to_pinned_origin() -> None:
    captured: dict[str, object] = {}

    def opener(request: object, *, timeout: float) -> object:
        captured["request"] = request
        captured["timeout"] = timeout
        raise HTTPError("http://api:8000/analyze", 502, "bad gateway", hdrs=Message(), fp=None)

    adapter = ApiAnalysisClient(
        base_url="http://api:8000",
        api_key="N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F",
        timeout_seconds=15,
        opener=opener,
    )
    with pytest.raises(DomainError) as error:
        adapter.analyze(
            FileAnalyzeInput(filename="ignored.wav", content_type="audio/wav", content=b"RIFF"),
            request_id=REQUEST_ID,
        )

    request = captured["request"]
    assert error.value.code is ErrorCode.AGENT_FAILURE
    assert request.full_url == "http://api:8000/analyze"  # type: ignore[union-attr]
    assert request.headers["Authorization"].startswith("Bearer ")  # type: ignore[union-attr]
    assert b'filename="audio"' in request.data  # type: ignore[union-attr]
    assert b"ignored.wav" not in request.data  # type: ignore[union-attr]


def test_pipeline_api_adapter_matches_rest_analysis_contract_with_local_stub() -> None:
    stub = LocalAnalyzeStub()
    adapter = ApiAnalysisClient(
        base_url="http://api:8000",
        api_key="N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F",
        timeout_seconds=15,
        opener=stub,
    )
    pipeline = Pipeline(analysis_adapter=adapter)
    verified = VerifiedAudio(
        name="authoritative.wav",
        content_type="audio/wav",
        content=b"RIFF\x00\x00\x00\x00WAVE",
    )

    rendered = pipeline._render_analysis(verified)

    request = stub.requests[0]
    assert request.full_url == "http://api:8000/analyze"  # type: ignore[union-attr]
    assert request.headers["Authorization"].startswith("Bearer ")  # type: ignore[union-attr]
    assert b"RIFF\x00\x00\x00\x00WAVE" in request.data  # type: ignore[union-attr]
    assert b"authoritative.wav" not in request.data  # type: ignore[union-attr]
    assert stub.timeouts == [15]
    assert "Клиент запросил информацию о кредите." in rendered
    assert "Отправить условия." in rendered
