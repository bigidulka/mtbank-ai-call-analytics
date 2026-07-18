from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mtbank_ai.speech.dataset import ManifestEntry
from scripts.evaluate_canonical_speech import CanonicalEvaluationFailure, _endpoint, _evaluate_entry


def _entry(tmp_path: Path) -> ManifestEntry:
    audio = tmp_path / "fixture.mp3"
    audio.write_bytes(b"fixture")
    return ManifestEntry(
        identifier="fixture",
        kind="speech_reference",
        root=tmp_path,
        path=audio,
        duration_seconds=1.0,
        sample_rate_hz=16000,
        channels=1,
        raw={"format": "mp3"},
    )


def test_canonical_evaluator_uses_fixed_transcribe_path_and_declared_mime(tmp_path: Path) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(502)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CanonicalEvaluationFailure, match="provider_failure") as error:
            _evaluate_entry(client, "http://speech.test/v1/transcribe", _entry(tmp_path))

    assert error.value.status_code == 502
    assert captured[0].url == httpx.URL("http://speech.test/v1/transcribe")
    assert b"Content-Type: audio/mpeg" in captured[0].content


@pytest.mark.parametrize("base_url", ("http://speech.test/prefix", "https://speech.test?next=x", "speech.test"))
def test_canonical_evaluator_rejects_noncanonical_base_url(base_url: str) -> None:
    with pytest.raises(ValueError):
        _endpoint(base_url)


def test_canonical_evaluator_reports_role_resolution_failure_without_parsing_body(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(409, text="do not store"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(CanonicalEvaluationFailure, match="role_resolution_required") as error:
            _evaluate_entry(client, "http://speech.test/v1/transcribe", _entry(tmp_path))

    assert error.value.status_code == 409
