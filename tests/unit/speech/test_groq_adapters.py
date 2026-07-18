from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from mtbank_ai.domain.transcript import WordTimestamp
from mtbank_ai.speech.contracts import AlignedSegment, DiarizationTurn
from services.speech.adapters import GroqWhisperTranscriber, LocalOverlapSpeakerAssigner
from services.speech.errors import SpeechProviderError
from services.speech.media import NormalizedAudio
from services.speech.settings import GroqTranscriptionSettings


class SyncChunks(httpx.SyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def _audio(tmp_path: Path) -> NormalizedAudio:
    path = tmp_path / "normalized.wav"
    path.write_bytes(b"RIFF")
    return NormalizedAudio(
        path=path,
        duration_seconds=2.0,
        audio_sha256="a" * 64,
        source_format="wav",
    )


def _settings(*, max_response_bytes: int = 4 * 1024 * 1024) -> GroqTranscriptionSettings:
    return GroqTranscriptionSettings(
        api_key=SecretStr("test-groq-key"),
        max_response_bytes=max_response_bytes,
    )


def _response(request: httpx.Request) -> httpx.Response:
    payload = json.dumps(
        {
            "text": "Добрый день клиент",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Добрый день"},
                {"start": 1.0, "end": 2.0, "text": "клиент"},
            ],
            "words": [
                {"word": "Добрый", "start": 0.0, "end": 0.4, "probability": 0.9},
                {"word": "день", "start": 0.4, "end": 1.0, "probability": 0.8},
                {"word": "клиент", "start": 1.0, "end": 1.5, "probability": 0.7},
            ],
            "usage": {"seconds": 1.5},
        },
        ensure_ascii=False,
    ).encode()
    return httpx.Response(
        200,
        headers={"x-request-id": "req_test_123"},
        stream=SyncChunks((payload,)),
        request=request,
    )


def test_groq_transcriber_sends_exact_verbose_multipart_once_and_maps_words(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(request)

    def client_factory(*, timeout: httpx.Timeout, trust_env: bool, follow_redirects: bool) -> httpx.Client:
        captured.update(timeout=timeout, trust_env=trust_env, follow_redirects=follow_redirects)
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )

    result = GroqWhisperTranscriber(_settings(), client_factory=client_factory).transcribe(
        _audio(tmp_path),
        language="ru",
    )

    assert len(requests) == 1
    assert requests[0].url == httpx.URL("https://api.groq.com/openai/v1/audio/transcriptions")
    assert requests[0].headers["authorization"] == "Bearer test-groq-key"
    assert requests[0].headers["accept-encoding"] == "identity"
    body = requests[0].content
    for name, value in (
        (b'model"', b"whisper-large-v3-turbo"),
        (b'language"', b"ru"),
        (b'temperature"', b"0"),
        (b'response_format"', b"verbose_json"),
        (b'timestamp_granularities[]"', b"segment"),
        (b'timestamp_granularities[]"', b"word"),
    ):
        assert name in body and value in body
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    assert result.segments[0].words[0].text == "Добрый"
    assert result.segments[1].words[0].text == "клиент"
    assert result.provider_metadata is not None
    assert result.provider_metadata.request_id == "req_test_123"
    assert result.provider_metadata.usage_seconds == 1.5


def test_groq_transcriber_bounds_response_and_never_falls_back(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-length": "17"},
            stream=SyncChunks((b"x" * 17,)),
            request=request,
        )

    def client_factory(*, timeout: httpx.Timeout, trust_env: bool, follow_redirects: bool) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            trust_env=trust_env,
            follow_redirects=follow_redirects,
        )

    transcriber = GroqWhisperTranscriber(_settings(max_response_bytes=16), client_factory=client_factory)
    with pytest.raises(SpeechProviderError) as error:
        transcriber.transcribe(_audio(tmp_path), language="ru")

    assert "test-groq-key" not in str(error.value)
    assert len(requests) == 1


def test_overlap_assigner_groups_consecutive_words_and_is_deterministic_at_edges() -> None:
    assigner = LocalOverlapSpeakerAssigner()
    aligned = (
        AlignedSegment(
            start=0.0,
            end=2.0,
            text="ignored",
            words=(
                WordTimestamp(word="один", start=0.0, end=0.5),
                WordTimestamp(word="два", start=0.5, end=1.0),
                WordTimestamp(word="три", start=0.9, end=1.1),
                WordTimestamp(word="четыре", start=1.1, end=1.2),
            ),
        ),
    )
    diarization = (
        DiarizationTurn(original_speaker_id="SPEAKER_A", start=0.0, end=1.0),
        DiarizationTurn(original_speaker_id="SPEAKER_B", start=1.0, end=2.0),
    )

    result = assigner.assign(aligned, diarization)

    assert [(segment.original_speaker_id, segment.text) for segment in result] == [
        ("SPEAKER_A", "один два три"),
        ("SPEAKER_B", "четыре"),
    ]
    assert result[0].speaker_confidence == pytest.approx((0.5 + 0.5 + 0.1) / (0.5 + 0.5 + 0.2))


def test_overlap_assigner_leaves_zero_overlap_unassigned_for_role_fail_closed() -> None:
    result = LocalOverlapSpeakerAssigner().assign(
        (
            AlignedSegment(
                start=2.0,
                end=2.1,
                text="слово",
                words=(WordTimestamp(word="слово", start=2.0, end=2.1),),
            ),
        ),
        (DiarizationTurn(original_speaker_id="SPEAKER_A", start=0.0, end=1.0),),
    )

    assert result[0].original_speaker_id is None
    assert result[0].speaker_confidence is None
