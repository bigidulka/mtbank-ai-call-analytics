from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

from mtbank_ai.api.dependencies import require_api_key
from mtbank_ai.api.main import create_app
from mtbank_ai.application.ports import AnalyzeInput, FileAnalyzeInput, UrlAnalyzeInput
from mtbank_ai.config import ApiSettings, DatabaseSettings, Settings
from mtbank_ai.domain.agents import ComplianceSeverity
from mtbank_ai.domain.analysis import (
    AnalysisMeta,
    AnalysisVersions,
    AnalyzeResponse,
    CompletedRunStatus,
    ComplianceView,
    GroundedActionItem,
    Grounding,
    PublicClassification,
    PublicComplianceIssue,
    PublicTranscriptSegment,
    QualityChecklist,
    QualityChecklistItem,
    QualityDetails,
    QualityScore,
)
from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.domain.provenance import ComponentRevision
from mtbank_ai.domain.transcript import SpeakerRole

REQUEST_ID = "11111111-1111-4111-8111-111111111111"
RUN_ID = UUID("22222222-2222-4222-8222-222222222222")
SEGMENT_ID = UUID("33333333-3333-4333-8333-333333333333")
SAFE_API_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"


class Ready:
    async def ping(self) -> bool:
        return True


class NotReady:
    async def ping(self) -> bool:
        return False


class StubAnalyzer:
    def __init__(self) -> None:
        self.sources: list[AnalyzeInput] = []
        self.request_ids: list[UUID] = []

    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        self.sources.append(source)
        self.request_ids.append(request_id)
        return _response()


class AgentFailureAnalyzer:
    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        del source, request_id
        raise DomainError(ErrorCode.AGENT_FAILURE)


class UnexpectedAnalyzer:
    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        del source, request_id
        raise RuntimeError("response-sentinel-must-not-leak")


class InvalidResponseAnalyzer:
    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        del source, request_id
        return cast(AnalyzeResponse, {"private": "response-sentinel-must-not-leak"})


def _settings(**api_changes: object) -> Settings:
    api = {"api_key": SecretStr(SAFE_API_KEY), **api_changes}
    return Settings(
        environment="test",
        api=ApiSettings.model_validate(api),
        database=DatabaseSettings(password=SecretStr("opaque-database-password")),
    )


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {SAFE_API_KEY}"}


def _item(*, passed: bool = True) -> QualityChecklistItem:
    return QualityChecklistItem(
        passed=passed,
        confidence=0.9,
        evidence_segment_ids=(SEGMENT_ID,),
        rationale="Критерий подтверждён.",
    )


def _response() -> AnalyzeResponse:
    return AnalyzeResponse(
        transcript=(
            PublicTranscriptSegment(
                id=SEGMENT_ID,
                speaker=SpeakerRole.OPERATOR,
                start=0.0,
                end=1.0,
                text="Добрый день.",
            ),
        ),
        classification=PublicClassification(
            topic="кредиты",
            priority="medium",
            confidence=0.9,
            evidence_segment_ids=(SEGMENT_ID,),
            rationale="Тема подтверждена.",
            taxonomy_version="taxonomy/v1",
        ),
        quality_score=QualityScore(
            total=75.0,
            checklist=QualityChecklist(
                greeting=True,
                need_detection=True,
                solution_provided=True,
                farewell=False,
            ),
            details=QualityDetails(
                greeting=_item(),
                need_detection=_item(),
                solution_provided=_item(),
                farewell=_item(passed=False),
            ),
            policy_version="quality/v1",
        ),
        compliance=ComplianceView(
            passed=True,
            issues=(
                PublicComplianceIssue(
                    rule_id="rule-1",
                    severity=ComplianceSeverity.INFO,
                    evidence_segment_ids=(SEGMENT_ID,),
                    explanation="Блокирующих нарушений нет.",
                ),
            ),
            policy_version="compliance/v1",
        ),
        summary="Клиент запросил информацию о кредите.",
        action_items=("Отправить условия.",),
        grounding=Grounding(
            summary_evidence_segment_ids=(SEGMENT_ID,),
            action_items=(GroundedActionItem(text="Отправить условия.", evidence_segment_ids=(SEGMENT_ID,)),),
        ),
        meta=AnalysisMeta(
            run_id=RUN_ID,
            status=CompletedRunStatus.COMPLETED,
            versions=AnalysisVersions(
                code_sha="abcdef0",
                prompt_bundle_hash="a" * 64,
                taxonomy_version="taxonomy/v1",
                quality_rubric_version="quality/v1",
                compliance_policy_version="compliance/v1",
                asr=ComponentRevision(
                    package="faster-whisper",
                    package_version="1.0.0",
                    model_id="large-v3",
                    model_revision="asr/v1",
                ),
                alignment=ComponentRevision(
                    package="whisperx",
                    package_version="3.0.0",
                    model_id="wav2vec2",
                    model_revision="alignment/v1",
                ),
                diarization=ComponentRevision(
                    package="pyannote.audio",
                    package_version="3.3.0",
                    model_id="speaker-diarization",
                    model_revision="diarization/v1",
                ),
            ),
            processing_ms=10,
            needs_review=False,
        ),
    )


async def _request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)  # type: ignore[arg-type]


def _error(response: httpx.Response) -> dict[str, object]:
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "request_id", "retryable"}
    return payload["error"]


def test_runtime_binding_is_bearer_protected_and_fails_closed_without_remote_speech() -> None:
    async def scenario() -> None:
        app = create_app(settings=_settings(), analyzer=StubAnalyzer(), readiness=Ready())

        assert (await _request(app, "GET", "/v1/benchmark-runtime-binding")).status_code == 401
        assert (await _request(app, "GET", "/v1/benchmark-runtime-binding", headers=_auth())).status_code == 503

    asyncio.run(scenario())


def test_health_and_openapi_are_public_but_analyze_requires_bearer_auth() -> None:
    async def scenario() -> None:
        app = create_app(settings=_settings(), analyzer=StubAnalyzer(), readiness=Ready())

        assert (await _request(app, "GET", "/health/live")).status_code == 200
        assert (await _request(app, "GET", "/openapi.json")).status_code == 200
        missing = await _request(app, "POST", "/analyze", json={"url": "https://example.test/call.wav"})
        wrong = await _request(
            app,
            "POST",
            "/analyze",
            json={"url": "https://example.test/call.wav"},
            headers={"Authorization": "Bearer wrong"},
        )
        correct = await _request(
            app,
            "POST",
            "/analyze",
            json={"url": "https://example.test/call.wav"},
            headers=_auth(),
        )

        assert missing.status_code == wrong.status_code == 401
        assert _error(missing)["code"] == _error(wrong)["code"] == "unauthenticated"
        assert correct.status_code == 200

    asyncio.run(scenario())


def test_auth_rejects_missing_and_non_ascii_credentials() -> None:
    async def scenario() -> None:
        settings = _settings()
        for credentials in (
            None,
            HTTPAuthorizationCredentials(scheme="Basic", credentials=SAFE_API_KEY),
            HTTPAuthorizationCredentials(scheme="Bearer", credentials="ключ"),
        ):
            with pytest.raises(DomainError) as error:
                await require_api_key(credentials, settings)
            assert error.value.code is ErrorCode.UNAUTHENTICATED

    asyncio.run(scenario())


def test_json_url_and_single_multipart_file_reach_injected_analyzer() -> None:
    async def scenario() -> None:
        analyzer = StubAnalyzer()
        app = create_app(settings=_settings(), analyzer=analyzer, readiness=Ready())
        json_response = await _request(
            app,
            "POST",
            "/analyze",
            json={"url": "http://example.test/call.wav"},
            headers=_auth(),
        )
        file_response = await _request(
            app,
            "POST",
            "/analyze",
            files={"file": ("call.wav", b"RIFF-valid-probe", "audio/wav")},
            headers={**_auth(), "X-Request-ID": REQUEST_ID},
        )

        assert json_response.status_code == file_response.status_code == 200
        assert isinstance(analyzer.sources[0], UrlAnalyzeInput)
        assert str(analyzer.sources[0].url) == "http://example.test/call.wav"
        assert isinstance(analyzer.sources[1], FileAnalyzeInput)
        assert analyzer.sources[1].content == b"RIFF-valid-probe"
        assert analyzer.request_ids[1] == UUID(REQUEST_ID)

    asyncio.run(scenario())


def test_source_and_request_error_taxonomy() -> None:
    async def scenario() -> None:
        app = create_app(settings=_settings(), analyzer=StubAnalyzer(), readiness=Ready())
        cases: tuple[tuple[dict[str, object], int, str], ...] = (
            ({"json": {}}, 400, "invalid_input"),
            ({"json": {"url": "https://example.test/a.wav", "other": 1}}, 422, "invalid_request"),
            ({"files": {"url": (None, "https://example.test/a.wav")}}, 422, "invalid_request"),
            (
                {"files": [("file", ("a.wav", b"RIFF", "audio/wav")), ("file", ("b.wav", b"RIFF", "audio/wav"))]},
                422,
                "invalid_request",
            ),
            ({"files": {"file": ("a.txt", b"text", "text/plain")}}, 415, "unsupported_media"),
            ({"content": b'{"url":', "headers": {"Content-Type": "application/json"}}, 422, "invalid_request"),
            ({"json": {"url": "not a URL"}}, 422, "invalid_url"),
            ({"files": {"file": ("call.wav", b"", "audio/wav")}}, 422, "invalid_audio"),
        )
        for request_kwargs, expected_status, expected_code in cases:
            headers = {**_auth(), **cast(dict[str, str], request_kwargs.pop("headers", {}))}
            response = await _request(app, "POST", "/analyze", headers=headers, **request_kwargs)
            assert response.status_code == expected_status
            assert _error(response)["code"] == expected_code

    asyncio.run(scenario())


def test_body_limits_reject_content_length_and_chunked_or_lying_bodies() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b'{"url":"https://example.test/'
        yield b"call.wav" + b"x" * 64 + b'"}'

    async def scenario() -> None:
        settings = _settings(max_json_bytes=32, max_upload_bytes=4, multipart_reserve_bytes=1024)
        app = create_app(settings=settings, analyzer=StubAnalyzer(), readiness=Ready())
        content_length = await _request(
            app,
            "POST",
            "/analyze",
            content=b"x" * 33,
            headers={**_auth(), "Content-Type": "application/json"},
        )
        chunked = await _request(
            app,
            "POST",
            "/analyze",
            content=chunks(),
            headers={**_auth(), "Content-Type": "application/json", "Transfer-Encoding": "chunked"},
        )
        lying_length = await _request(
            app,
            "POST",
            "/analyze",
            content=b"x" * 33,
            headers={**_auth(), "Content-Type": "application/json", "Content-Length": "1"},
        )
        oversized_file = await _request(
            app,
            "POST",
            "/analyze",
            files={"file": ("call.wav", b"12345", "audio/wav")},
            headers=_auth(),
        )

        for response in (content_length, chunked, lying_length, oversized_file):
            assert response.status_code == 413
            assert _error(response)["code"] == "payload_too_large"

    asyncio.run(scenario())


def test_body_limiter_covers_slash_alias_without_changing_redirect_semantics() -> None:
    async def scenario() -> None:
        app = create_app(
            settings=_settings(max_json_bytes=32, max_upload_bytes=4, multipart_reserve_bytes=1024),
            analyzer=StubAnalyzer(),
            readiness=Ready(),
        )
        redirect = await _request(
            app,
            "POST",
            "/analyze/",
            content=b"{}",
            headers={**_auth(), "Content-Type": "application/json"},
        )
        oversized = await _request(
            app,
            "POST",
            "/analyze/",
            content=b"x" * 33,
            headers={**_auth(), "Content-Type": "application/json"},
        )

        assert redirect.status_code == 307
        assert redirect.headers["location"].endswith("/analyze")
        assert oversized.status_code == 413
        assert "location" not in oversized.headers
        assert _error(oversized)["code"] == "payload_too_large"

    asyncio.run(scenario())


def test_unexpected_analyzer_and_response_validation_errors_hide_sentinels() -> None:
    async def scenario() -> None:
        unexpected_app = create_app(settings=_settings(), analyzer=UnexpectedAnalyzer(), readiness=Ready())
        invalid_response_app = create_app(settings=_settings(), analyzer=InvalidResponseAnalyzer(), readiness=Ready())
        unexpected = await _request(
            unexpected_app,
            "POST",
            "/analyze",
            json={"url": "https://example.test/call.wav"},
            headers=_auth(),
        )
        invalid_response = await _request(
            invalid_response_app,
            "POST",
            "/analyze",
            json={"url": "https://example.test/call.wav"},
            headers=_auth(),
        )

        for response in (unexpected, invalid_response):
            assert response.status_code == 500
            assert _error(response)["code"] == "internal_error"
            assert "response-sentinel-must-not-leak" not in response.text

    asyncio.run(scenario())


def test_native_404_and_405_semantics_and_agent_failure_are_preserved() -> None:
    async def scenario() -> None:
        app = create_app(settings=_settings(), analyzer=AgentFailureAnalyzer(), readiness=NotReady())
        missing = await _request(app, "GET", "/missing")
        method = await _request(app, "GET", "/analyze")
        agent_failure = await _request(
            app,
            "POST",
            "/analyze",
            json={"url": "https://example.test/call.wav"},
            headers=_auth(),
        )
        ready = await _request(app, "GET", "/health/ready")

        assert missing.status_code == 404 and missing.json() == {"detail": "Not Found"}
        assert method.status_code == 405
        assert method.json() == {"detail": "Method Not Allowed"}
        assert method.headers["allow"] == "POST"
        assert agent_failure.status_code == 502
        assert _error(agent_failure)["code"] == "agent_failure"
        assert ready.status_code == 503

    asyncio.run(scenario())


def test_openapi_is_exact_for_media_types_bearer_security_and_public_response() -> None:
    async def scenario() -> None:
        app = create_app(settings=_settings(), analyzer=StubAnalyzer(), readiness=Ready())
        schema = (await _request(app, "GET", "/openapi.json")).json()
        operation = schema["paths"]["/analyze"]["post"]
        multipart = operation["requestBody"]["content"]["multipart/form-data"]["schema"]

        assert schema["components"]["securitySchemes"]["BearerAuth"] == {"type": "http", "scheme": "bearer"}
        assert operation["security"] == [{"BearerAuth": []}]
        assert set(operation["requestBody"]["content"]) == {"multipart/form-data", "application/json"}
        assert multipart == {
            "type": "object",
            "additionalProperties": False,
            "required": ["file"],
            "properties": {"file": {"type": "string", "format": "binary"}},
        }
        assert operation["requestBody"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/UrlAnalyzeRequest"
        }
        assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/AnalyzeResponse"
        )
        assert operation["responses"]["401"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/ErrorResponse"
        )

    asyncio.run(scenario())
