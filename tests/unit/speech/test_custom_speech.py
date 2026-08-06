from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
import pytest
from pydantic import HttpUrl, SecretStr, TypeAdapter

from mtbank_ai.domain.transcript import RoleAgentProvenance, SpeakerRole
from mtbank_ai.speech.contracts import SpeechFile
from services.custom_speech.runtime import (
    FlatTurn,
    OpenAiFlatTranscriber,
    SemanticTurnResolver,
    SemanticVadSpeechRuntime,
    VadAnchor,
    _align_turns,
    _snapshot,
    _validate_turn_coverage,
)
from services.custom_speech.settings import CustomSpeechSettings
from services.speech.errors import SpeechProviderError
from services.speech.media import NormalizedAudio


def _settings() -> CustomSpeechSettings:
    return CustomSpeechSettings(
        asr_api_key=SecretStr("A" * 32 + "openai"),
        role_base_url=TypeAdapter(HttpUrl).validate_python("https://llm.example.test/v1"),
        role_api_key=SecretStr("B" * 32 + "gateway"),
        role_model="gpt-5.6-sol",
    )


def _turn(
    start: int,
    end: int,
    role: Literal["Оператор", "Клиент"],
    confidence: float = 0.9,
) -> FlatTurn:
    return FlatTurn(
        start_word_index=start,
        end_word_index=end,
        role=role,
        confidence=confidence,
        evidence="test evidence",
    )


def test_openai_flat_transcriber_sends_one_bounded_allowlisted_request(tmp_path: Path) -> None:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"RIFF-test")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://api.openai.com/v1/audio/transcriptions"
        assert request.headers["authorization"].startswith("Bearer ")
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b'{"text":"hello"}'),
            headers={"x-request-id": "request-1", "content-encoding": "identity"},
        )

    transcriber = OpenAiFlatTranscriber(
        _settings(),
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = transcriber.transcribe(
        NormalizedAudio(
            path=audio_path,
            duration_seconds=1.0,
            audio_sha256="a" * 64,
            source_format="wav",
        ),
        language="ru",
    )

    assert len(requests) == 1
    assert result.text == "hello"
    assert result.provider_metadata.provider == "openai"
    assert result.provider_metadata.request_id == "request-1"


def test_bridge_transcription_is_attested_as_the_bridge_not_an_official_provider(tmp_path: Path) -> None:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"RIFF-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://chatgpt-bridge:37182/v1/audio/transcriptions"
        return httpx.Response(200, stream=httpx.ByteStream(b'{"text":"hello"}'))

    settings = _settings().model_copy(
        update={
            "asr_endpoint": TypeAdapter(HttpUrl).validate_python("http://chatgpt-bridge:37182/v1/audio/transcriptions"),
            "pipeline_revision": "speech/chatgpt-bridge-semantic-vad-v1",
        }
    )
    transcriber = OpenAiFlatTranscriber(
        settings,
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs),
    )
    result = transcriber.transcribe(
        NormalizedAudio(
            path=audio_path,
            duration_seconds=1.0,
            audio_sha256="a" * 64,
            source_format="wav",
        ),
        language="ru",
    )

    assert result.provider_metadata.provider == "chatgpt-bridge"


def _role_response(turns: object) -> httpx.Response:
    body = {
        "model": "gpt-5.6-sol",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "submit_flat_turns",
                                "arguments": json.dumps({"turns": turns}, ensure_ascii=False),
                            },
                        }
                    ]
                }
            }
        ],
    }
    return httpx.Response(200, stream=httpx.ByteStream(json.dumps(body, ensure_ascii=False).encode()))


def test_semantic_role_resolver_retries_a_transient_malformed_decision() -> None:
    good = [
        {
            "start_word_index": 0,
            "end_word_index": 1,
            "role": "Оператор",
            "confidence": 0.9,
            "evidence": "test evidence",
        }
    ]
    responses = [_role_response([20]), _role_response(good)]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    resolver = SemanticTurnResolver(
        _settings(),
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs),
    )

    turns = resolver.resolve(("добрый", "день"))

    assert [turn.role for turn in turns] == ["Оператор"]
    assert not responses


def test_semantic_role_resolver_stops_at_the_attempt_budget() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return _role_response([20])

    settings = _settings().model_copy(update={"role_max_attempts": 2})
    resolver = SemanticTurnResolver(
        settings,
        client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handler), **kwargs),
    )

    with pytest.raises(SpeechProviderError, match="semantic role request failed"):
        resolver.resolve(("добрый", "день"))

    assert attempts == 2


def test_adaptive_vad_alignment_reduces_padding_without_overlap() -> None:
    turns = (
        _turn(0, 1, "Оператор"),
        _turn(2, 3, "Клиент"),
    )
    anchors = (VadAnchor(start=0.5, end=1.0), VadAnchor(start=1.3, end=2.0))

    aligned = _align_turns(turns, ("добрый", "день", "нужна", "помощь"), anchors, 3.0)

    assert [(item.start, item.end) for item in aligned] == [(0.4, 1.2), (1.2, 2.2)]
    assert [item.role for item in aligned] == ["Оператор", "Клиент"]
    assert [item.text for item in aligned] == ["добрый день", "нужна помощь"]


def test_vad_alignment_fails_closed_when_turn_topology_exceeds_anchors() -> None:
    turns = (_turn(0, 0, "Оператор"), _turn(1, 1, "Клиент"))

    with pytest.raises(SpeechProviderError, match="exceed VAD anchor topology"):
        _align_turns(turns, ("да", "нет"), (VadAnchor(start=0.0, end=1.0),), 1.0)


def test_semantic_turn_coverage_rejects_gap() -> None:
    with pytest.raises(ValueError, match="gap, overlap, or reorder"):
        _validate_turn_coverage((_turn(1, 1, "Клиент"),), 2)


def test_custom_snapshot_preserves_canonical_contract(tmp_path: Path) -> None:
    settings = _settings()
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"RIFF-test")
    audio = NormalizedAudio(
        path=audio_path,
        duration_seconds=3.0,
        audio_sha256="a" * 64,
        source_format="wav",
    )
    turns = (_turn(0, 1, "Оператор"), _turn(2, 3, "Клиент", 0.7))
    aligned = _align_turns(
        turns,
        ("добрый", "день", "нужна", "помощь"),
        (VadAnchor(start=0.2, end=1.0), VadAnchor(start=1.5, end=2.5)),
        3.0,
    )

    snapshot = _snapshot(
        source=SpeechFile(filename="call.wav", content_type="audio/wav", content=b"RIFF-test"),
        audio=audio,
        turns=turns,
        aligned=aligned,
        provider=None,
        role_provenance=RoleAgentProvenance(
            policy_id="flat_transcript_reconstructor",
            version="v1",
            owner="MTBank AI Engineering",
            effective_date="2026-08-04",
            sha256="b" * 64,
        ),
        pipeline_revision=settings.pipeline_revision,
        processing_ms=123,
        settings=settings,
    )

    assert snapshot.revision == "speech/openai-semantic-vad-v1"
    assert [segment.speaker for segment in snapshot.segments] == [SpeakerRole.OPERATOR, SpeakerRole.CLIENT]
    assert snapshot.role_resolution.needs_review is True
    assert snapshot.asr_metadata.asr.model_id == "gpt-4o-transcribe"
    assert snapshot.asr_metadata.diarization.model_id == "semantic-vad/gpt-5.6-sol"


def test_snapshot_survives_a_role_whose_turns_disagree_on_confidence(tmp_path: Path) -> None:
    settings = _settings()
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"RIFF-test")
    audio = NormalizedAudio(
        path=audio_path,
        duration_seconds=6.0,
        audio_sha256="a" * 64,
        source_format="wav",
    )
    # The operator speaks twice with different certainty; the canonical contract binds one
    # confidence per role, so both operator segments must report the role's weakest value.
    turns = (
        _turn(0, 1, "Оператор", 0.9),
        _turn(2, 3, "Клиент", 0.8),
        _turn(4, 5, "Оператор", 0.4),
    )
    words = ("добрый", "день", "нужна", "помощь", "уточняю", "детали")
    anchors = (
        VadAnchor(start=0.2, end=1.0),
        VadAnchor(start=1.5, end=2.5),
        VadAnchor(start=3.0, end=4.5),
    )
    aligned = _align_turns(turns, words, anchors, 6.0)

    snapshot = _snapshot(
        source=SpeechFile(filename="call.wav", content_type="audio/wav", content=b"RIFF-test"),
        audio=audio,
        turns=turns,
        aligned=aligned,
        provider=None,
        role_provenance=RoleAgentProvenance(
            policy_id="flat_transcript_reconstructor",
            version="v1",
            owner="MTBank AI Engineering",
            effective_date="2026-08-04",
            sha256="b" * 64,
        ),
        pipeline_revision=settings.pipeline_revision,
        processing_ms=1,
        settings=settings,
    )

    operator = [segment for segment in snapshot.segments if segment.speaker is SpeakerRole.OPERATOR]
    assert [segment.role_confidence for segment in operator] == [0.4, 0.4]
    assert [segment.speaker_confidence for segment in operator] == [0.9, 0.4]


def test_custom_runtime_attestation_identifies_experimental_profile() -> None:
    runtime = SemanticVadSpeechRuntime.__new__(SemanticVadSpeechRuntime)
    runtime._settings = _settings()  # pyright: ignore[reportPrivateUsage]

    attestation = runtime.runtime_attestation()

    assert attestation["runtime"]["profile"] == "experimental_no_gpu"  # type: ignore[index]
    assert attestation["runtime"]["role_model"] == "gpt-5.6-sol"  # type: ignore[index]
