from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import HttpUrl, SecretStr

from mtbank_ai.agent_runtime import ModelResponse, ModelToolCall, ModelUsage
from mtbank_ai.domain.transcript import SpeakerRole
from mtbank_ai.speech.contracts import RoleResolutionCandidate, RoleSegmentEvidence
from services.speech.errors import SpeechProviderError
from services.speech.role_agent import LlmRoleResolver
from services.speech.settings import RoleAgentSettings

OPERATOR_ID = UUID("11111111-1111-4111-8111-111111111111")
CLIENT_ID = UUID("22222222-2222-4222-8222-222222222222")


class Provider:
    def __init__(self, response: ModelResponse, *additional_responses: ModelResponse) -> None:
        self.responses = [response, *additional_responses]
        self.requests = []
        self.deadlines = []

    def complete(self, request, *, deadline_at):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        self.deadlines.append(deadline_at)
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def _settings() -> RoleAgentSettings:
    return RoleAgentSettings(
        base_url=HttpUrl("https://gateway.example.test/v1"),
        api_key=SecretStr("N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"),
        model="role-model",
    )


def _candidates() -> tuple[RoleResolutionCandidate, ...]:
    return (
        RoleResolutionCandidate(
            original_speaker_id="speaker-a",
            evidence_segment_ids=(OPERATOR_ID,),
            evidence_segments=(RoleSegmentEvidence(segment_id=OPERATOR_ID, text="Добрый день, слушаю вас."),),
        ),
        RoleResolutionCandidate(
            original_speaker_id="speaker-b",
            evidence_segment_ids=(CLIENT_ID,),
            evidence_segments=(RoleSegmentEvidence(segment_id=CLIENT_ID, text="Хочу уточнить перевод."),),
        ),
    )


def _response(arguments: object, *, model: str = "role-model", text: bool = False) -> ModelResponse:
    return ModelResponse(
        request_id="request-1",
        model_id=model,
        finish_reason="tool_calls",
        tool_calls=(
            ModelToolCall(
                id="call-1",
                name="submit_role_resolution",
                arguments_json=json.dumps(arguments),
            ),
        ),
        usage=ModelUsage(input_tokens=100, output_tokens=40, total_tokens=140),
        latency_ms=10,
        has_text_content=text,
        text_content="not allowed" if text else None,
    )


def test_role_agent_uses_one_required_terminal_tool_and_returns_grounded_roles() -> None:
    provider = Provider(
        _response(
            {
                "roles": [
                    {
                        "original_speaker_id": "speaker-a",
                        "role": "Оператор",
                        "confidence": 0.91,
                        "evidence": "model-grounded",
                        "evidence_segment_ids": [str(OPERATOR_ID)],
                    },
                    {
                        "original_speaker_id": "speaker-b",
                        "role": "Клиент",
                        "confidence": 0.88,
                        "evidence": "model-grounded",
                        "evidence_segment_ids": [str(CLIENT_ID)],
                    },
                ],
                "agent_provenance": None,
            }
        )
    )

    decision = LlmRoleResolver(_settings(), provider=provider).resolve(_candidates())

    assert {item.role for item in decision.roles} == {SpeakerRole.OPERATOR, SpeakerRole.CLIENT}
    assert decision.agent_provenance is not None
    assert decision.agent_provenance.policy_id == "role_resolver"
    request = provider.requests[0]
    assert request.model_id == "role-model"
    assert request.tool_choice.value == "required"
    assert len(request.tools) == 1
    assert request.tools[0].name == "submit_role_resolution"
    assert request.temperature == 0.0
    assert "`evidence` должен быть непустой краткой причиной" in request.messages[0].content
    assert "ровно один самый сильный точный segment ID" in request.messages[0].content


@pytest.mark.parametrize(
    "response",
    (
        _response({"roles": [], "agent_provenance": None}, model="other-model"),
        _response({"roles": [], "agent_provenance": None}, text=True),
    ),
)
def test_role_agent_rejects_model_drift_and_text_completion(response: ModelResponse) -> None:
    with pytest.raises(SpeechProviderError, match="invalid terminal response"):
        LlmRoleResolver(_settings(), provider=Provider(response)).resolve(_candidates())


def test_role_agent_retries_once_when_provider_returns_invalid_typed_output() -> None:
    valid = _response(
        {
            "roles": [
                {
                    "original_speaker_id": "speaker-a",
                    "role": "Оператор",
                    "confidence": 0.91,
                    "evidence": "приветствие оператора",
                    "evidence_segment_ids": [str(OPERATOR_ID)],
                },
                {
                    "original_speaker_id": "speaker-b",
                    "role": "Клиент",
                    "confidence": 0.88,
                    "evidence": "запрос клиента",
                    "evidence_segment_ids": [str(CLIENT_ID)],
                },
            ],
            "agent_provenance": None,
        }
    )
    provider = Provider(_response({"roles": [{"original_speaker_id": "speaker-a"}]}), valid)

    decision = LlmRoleResolver(_settings(), provider=provider).resolve(_candidates())

    assert len(decision.roles) == 2
    assert len(provider.requests) == 2
    assert "Предыдущий JSON не прошёл схему" in provider.requests[1].messages[-1].content


def test_role_agent_rejects_second_invalid_typed_output() -> None:
    invalid = _response({"roles": [{"original_speaker_id": "speaker-a"}]})
    provider = Provider(invalid, invalid)

    with pytest.raises(SpeechProviderError, match="invalid typed output"):
        LlmRoleResolver(_settings(), provider=provider).resolve(_candidates())

    assert len(provider.requests) == 2


def test_role_agent_returns_empty_typed_decision_when_input_exceeds_bound() -> None:
    settings = _settings().model_copy(update={"max_input_chars": 10})
    provider = Provider(_response({"roles": [], "agent_provenance": None}))

    decision = LlmRoleResolver(settings, provider=provider).resolve(_candidates())

    assert decision.roles == ()
    assert provider.requests == []
