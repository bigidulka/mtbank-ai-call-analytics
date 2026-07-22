from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import pytest

import scripts.evaluate_canonical_speech as canonical_evaluator
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


def test_canonical_evaluator_bearer_mode_requires_safe_https_and_one_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"
    monkeypatch.setenv("CANONICAL_TEST_KEY", key)
    headers = canonical_evaluator._bearer_headers("CANONICAL_TEST_KEY")
    assert headers == {"Authorization": f"Bearer {key}"}
    assert canonical_evaluator._endpoint("https://speech.test", bearer=True) == "https://speech.test/v1/transcribe"
    for unsafe in ("http://speech.test", "https://key@speech.test", "https://speech.test?x=1"):
        with pytest.raises(ValueError) as error:
            canonical_evaluator._endpoint(unsafe, bearer=True)
        assert key not in str(error.value)

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(502)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(CanonicalEvaluationFailure):
            _evaluate_entry(client, "https://speech.test/v1/transcribe", _entry(tmp_path), headers)
    assert captured[0].headers.get_list("authorization") == [f"Bearer {key}"]


def test_canonical_evaluator_disables_environment_proxies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("fixtures: []\n", encoding="utf-8")
    client_options: dict[str, object] = {}

    class Client:
        def __init__(self, **kwargs: object) -> None:
            client_options.update(kwargs)

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(canonical_evaluator, "validate_manifest", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(canonical_evaluator.httpx, "Client", Client)

    status, result = canonical_evaluator.evaluate(
        argparse.Namespace(manifest=manifest, base_url="http://speech.test", timeout_seconds=12.5)
    )

    assert status == 0
    assert result["status"] == "completed"
    assert client_options == {"timeout": 12.5, "follow_redirects": False, "trust_env": False}


def test_canonical_evaluator_reports_role_resolution_failure_without_parsing_body(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(409, text="do not store"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(CanonicalEvaluationFailure, match="role_resolution_required") as error:
            _evaluate_entry(client, "http://speech.test/v1/transcribe", _entry(tmp_path))

    assert error.value.status_code == 409
