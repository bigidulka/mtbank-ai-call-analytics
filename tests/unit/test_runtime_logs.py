from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).parents[2]


def _contains_sensitive_marker() -> Callable[[str], bool]:
    namespace = runpy.run_path(str(ROOT / "scripts" / "assert_runtime_logs_clean.py"))
    return cast(Callable[[str], bool], namespace["_contains_sensitive_marker"])


def _contains_access_request_line() -> Callable[[str], bool]:
    namespace = runpy.run_path(str(ROOT / "scripts" / "assert_runtime_logs_clean.py"))
    return cast(Callable[[str], bool], namespace["_contains_access_request_line"])


def _contains_nginx_error_request_line() -> Callable[[str], bool]:
    namespace = runpy.run_path(str(ROOT / "scripts" / "assert_runtime_logs_clean.py"))
    return cast(Callable[[str], bool], namespace["_contains_nginx_error_request_line"])


def test_runtime_log_guard_accepts_warning_only_output() -> None:
    logs = "openwebui | WARNING retry disabled\npipelines | ERROR upstream unavailable"

    assert not _contains_sensitive_marker()(logs)
    assert not _contains_access_request_line()(logs)
    assert not _contains_nginx_error_request_line()(logs)


@pytest.mark.parametrize(
    "logs",
    [
        "Authorization: Bearer secret",
        "Pipeline input: {'messages': []}",
        "Проверь приложенный WAV.",
        "uploaded attachment-positive.wav",
        "blocked image_url https://example.invalid/unbounded-image",
        "filename contains https:attacker.invalid",
    ],
)
def test_runtime_log_guard_rejects_payload_auth_and_uri_markers(logs: str) -> None:
    assert _contains_sensitive_marker()(logs)


def test_runtime_log_guard_rejects_nginx_access_request_lines() -> None:
    logs = (
        "gateway-1 | 172.20.0.1 - - [15/Jul/2026:12:00:00 +0000] "
        '"POST /api/v1/files/?process=false HTTP/1.1" 413 183 '
        '"https://app.example.test/chat?id=private" "Python-urllib/3.13"'
    )

    assert _contains_access_request_line()(logs)


def test_runtime_log_guard_rejects_nginx_error_request_metadata() -> None:
    logs = (
        "gateway-1 | 2026/07/15 12:00:00 [error] 29#29: *7 client intended to send too large body: "
        '26279937 bytes, client: 172.20.0.1, server: _, request: "POST '
        '/api/v1/files/?process=false&attachment=private HTTP/1.1", host: "localhost:3000"'
    )

    assert _contains_nginx_error_request_line()(logs)
