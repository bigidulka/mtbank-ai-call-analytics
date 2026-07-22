from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import scripts.run_gpu_speech_benchmark as gpu_benchmark
import scripts.run_local_speech_sla_benchmark as local_benchmark
import scripts.run_public_analyze_sla_benchmark as public_benchmark

_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"
_DIGEST = "sha256:" + "a" * 64


def test_local_sla_bearer_mode_requires_safe_https_and_one_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_BENCHMARK_KEY", _KEY)

    assert local_benchmark._bearer_headers("LOCAL_BENCHMARK_KEY") == {"Authorization": f"Bearer {_KEY}"}
    assert local_benchmark._endpoint("https://speech.test", bearer=True) == "https://speech.test/v1/transcribe"
    for unsafe in ("http://speech.test", "https://key@speech.test", "https://speech.test#fragment"):
        with pytest.raises(ValueError) as error:
            local_benchmark._endpoint(unsafe, bearer=True)
        assert _KEY not in str(error.value)


def test_gpu_benchmark_requires_remote_attested_digest_and_sends_one_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_BENCHMARK_KEY", _KEY)
    captured: list[httpx.Request] = []
    payload = {
        "runtime": {
            "device": "cuda",
            "compute_type": "float16",
            "image_digest": _DIGEST,
            "asr": {"package": "faster-whisper", "package_version": "1", "model_id": "asr", "model_revision": "r1"},
            "diarization": {
                "package": "pyannote.audio",
                "package_version": "1",
                "model_id": "diarization",
                "model_revision": "r2",
            },
        }
    }

    class Client:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {"timeout": 15.0, "follow_redirects": False, "trust_env": False}

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
            request = httpx.Request("GET", url, headers=headers)
            captured.append(request)
            return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(gpu_benchmark.httpx, "Client", Client)
    attestation_hash = gpu_benchmark._runtime_attestation(
        "https://speech.test/v1/runtime", "GPU_BENCHMARK_KEY", _DIGEST
    )

    assert len(attestation_hash) == 64
    assert captured[0].headers.get_list("authorization") == [f"Bearer {_KEY}"]
    payload["runtime"]["image_digest"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="image digest"):
        gpu_benchmark._runtime_attestation("https://speech.test/v1/runtime", "GPU_BENCHMARK_KEY", _DIGEST)


def test_public_analyze_runner_requires_https_and_records_no_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUBLIC_BENCHMARK_KEY", _KEY)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")

    def make_workload(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"workload")

    class Client:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["follow_redirects"] is False and kwargs["trust_env"] is False

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, *, files: object, headers: dict[str, str]) -> httpx.Response:
            del files
            assert url == "https://api.test/analyze"
            assert headers == {"Authorization": f"Bearer {_KEY}"}
            return httpx.Response(200, content=b"private response is never recorded")

    analyzed = SimpleNamespace(
        meta=SimpleNamespace(
            versions=SimpleNamespace(
                code_sha="abc",
                asr=SimpleNamespace(model_revision="asr-r1"),
                diarization=SimpleNamespace(model_revision="diarization-r1"),
            )
        )
    )
    monkeypatch.setattr(public_benchmark, "_make_five_minutes", make_workload)
    monkeypatch.setattr(public_benchmark, "_duration_seconds", lambda _path: 300.0)
    monkeypatch.setattr(public_benchmark.httpx, "Client", Client)
    monkeypatch.setattr(public_benchmark.AnalyzeResponse, "model_validate_json", lambda _content: analyzed)
    arguments = argparse.Namespace(
        base_url="https://api.test",
        api_key_env="PUBLIC_BENCHMARK_KEY",
        audio=source,
        timeout_seconds=1.0,
    )

    status, result = public_benchmark.run(arguments)

    assert status == 0
    assert result["audio_seconds"] == 300.0
    assert "private response" not in str(result)
    with pytest.raises(ValueError):
        public_benchmark._endpoint("http://api.test")
