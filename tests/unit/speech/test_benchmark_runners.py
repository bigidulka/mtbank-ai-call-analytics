from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import mtbank_ai.public_endpoint as public_endpoint
import scripts.run_gpu_speech_benchmark as gpu_benchmark
import scripts.run_local_speech_sla_benchmark as local_benchmark
import scripts.run_public_analyze_sla_benchmark as public_benchmark
import scripts.run_websocket_benchmark as websocket_benchmark
from mtbank_ai.release.model_manifest import ModelArtifact, SpeechModelManifest

_KEY = "N7!qR2@vL9#sX4$kM8%tY1^cD6&hJ3*F"
_DIGEST = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        public_endpoint.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("8.8.8.8", port))],
    )


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
        "schema_version": "1",
        "kind": "configured-remote-speech-runtime",
        "speech_backend_url_sha256": "b" * 64,
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
        },
    }
    model_manifest = SpeechModelManifest(
        asr=ModelArtifact(
            package="faster-whisper",
            package_version="1",
            model_id="asr",
            model_revision="r1",
            relative_path="asr",
            artifact_sha256="a" * 64,
        ),
        diarization=ModelArtifact(
            package="pyannote.audio",
            package_version="1",
            model_id="diarization",
            model_revision="r2",
            relative_path="diarization",
            artifact_sha256="b" * 64,
        ),
    )

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
    attestation_hash = gpu_benchmark._runtime_binding(
        "https://api.test/v1/benchmark-runtime-binding", "GPU_BENCHMARK_KEY", _DIGEST, model_manifest
    )

    assert len(attestation_hash) == 64
    assert captured[0].headers.get_list("authorization") == [f"Bearer {_KEY}"]
    payload["runtime"]["image_digest"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="image digest"):
        gpu_benchmark._runtime_binding(
            "https://api.test/v1/benchmark-runtime-binding", "GPU_BENCHMARK_KEY", _DIGEST, model_manifest
        )
    payload["runtime"]["image_digest"] = _DIGEST
    payload["runtime"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CUDA float16"):
        gpu_benchmark._runtime_binding(
            "https://api.test/v1/benchmark-runtime-binding", "GPU_BENCHMARK_KEY", _DIGEST, model_manifest
        )
    payload["runtime"]["device"] = "cuda"
    payload["runtime"]["compute_type"] = "int8"
    with pytest.raises(ValueError, match="CUDA float16"):
        gpu_benchmark._runtime_binding(
            "https://api.test/v1/benchmark-runtime-binding", "GPU_BENCHMARK_KEY", _DIGEST, model_manifest
        )
    payload["runtime"]["compute_type"] = "float16"
    payload["runtime"]["asr"]["model_revision"] = "unexpected"
    with pytest.raises(ValueError, match="asr revision"):
        gpu_benchmark._runtime_binding(
            "https://api.test/v1/benchmark-runtime-binding", "GPU_BENCHMARK_KEY", _DIGEST, model_manifest
        )


def test_gpu_benchmark_derives_runtime_binding_from_app_wss_authority() -> None:
    assert gpu_benchmark._runtime_binding_url("wss://api.example.test/ws/transcribe") == (
        "https://api.example.test/v1/benchmark-runtime-binding"
    )
    for websocket_url in (
        "ws://api.example.test/ws/transcribe",
        "wss://api.example.test/v1/stream",
        "wss://127.0.0.1/ws/transcribe",
        "wss://api.internal/ws/transcribe",
    ):
        with pytest.raises(ValueError):
            gpu_benchmark._runtime_binding_url(websocket_url)


def test_credential_endpoints_reject_dns_failure_and_non_public_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(public_endpoint.socket, "getaddrinfo", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="no addresses"):
        websocket_benchmark._validate_url("wss://api.example.test/ws/transcribe")

    monkeypatch.setattr(
        public_endpoint.socket,
        "getaddrinfo",
        lambda _host, port, **_kwargs: [(2, 1, 6, "", ("127.0.0.1", port))],
    )
    with pytest.raises(ValueError, match="non-public"):
        public_benchmark._endpoint("https://api.example.test")


def test_websocket_benchmark_requires_wss_and_rejects_redirect_before_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WS_BENCHMARK_KEY", _KEY)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 8_000)
    captured: dict[str, object] = {}

    class RedirectedHandshake:
        async def __aenter__(self) -> object:
            raise RuntimeError("handshake redirect rejected")

        async def __aexit__(self, *_args: object) -> None:
            return None

    def connect(*args: object, **kwargs: object) -> RedirectedHandshake:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return RedirectedHandshake()

    monkeypatch.setattr(websocket_benchmark, "_connect_without_redirects", connect)
    arguments = argparse.Namespace(
        url="wss://api.example.test/ws/transcribe",
        origin="https://web.example.test",
        audio=audio,
        api_key_env="WS_BENCHMARK_KEY",
        frame_ms=500,
        response_timeout_seconds=1.0,
        max_message_bytes=98_304,
    )

    with pytest.raises(RuntimeError, match="redirect rejected"):
        asyncio.run(websocket_benchmark.run(arguments))

    assert captured["args"] == ("wss://api.example.test/ws/transcribe",)
    assert captured["kwargs"] == {
        "origin": "https://web.example.test",
        "additional_headers": [("Authorization", f"Bearer {_KEY}")],
        "compression": None,
        "max_size": 98_304,
        "proxy": None,
    }
    with pytest.raises(ValueError, match="WSS"):
        websocket_benchmark._validate_url("ws://api.example.test/ws/transcribe")


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
