"""Проверяет host-direct pre-body boundary закреплённого OpenWebUI wrapper."""

from __future__ import annotations

import argparse
import http.client
import json
import time
from urllib.parse import urlsplit

MAX_REQUEST_BODY_BYTES = 25 * 1024 * 1024 + 64 * 1024
_IMMEDIATE_RESPONSE_SECONDS = 3
_DENIED_RESOURCE_ROUTES = (
    "/api/v1/retrieval/process/file",
    "/api/v1/retrieval/process/files/batch",
    "/api/v1/retrieval/process/text",
    "/api/v1/retrieval/process/web",
    "/api/v1/retrieval/process/youtube",
    "/api/v1/retrieval/process/web/search",
    "/api/v1/files/attachment-id/data/content/update",
    "/api/v1/knowledge/knowledge-id/file/add",
    "/api/v1/knowledge/knowledge-id/file/update",
    "/api/v1/knowledge/knowledge-id/files/batch/add",
    "/api/v1/audio/transcriptions",
    "/api/v1/audio/speech",
)


def _connection(base_url: str) -> http.client.HTTPConnection:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("--base-url должен быть простым HTTP base URL")
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=_IMMEDIATE_RESPONSE_SECONDS)


def _headers_only_request(base_url: str, path: str) -> tuple[int, str, float]:
    connection = _connection(base_url)
    started = time.monotonic()
    try:
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
        connection.endheaders()
        response = connection.getresponse()
        detail = response.read(400).decode("utf-8", errors="replace")
    finally:
        connection.close()
    return response.status, detail, time.monotonic() - started


def _assert_immediate_status(base_url: str, path: str, expected_status: int, detail: str) -> float:
    status, response_body, elapsed = _headers_only_request(base_url, path)
    if status != expected_status or detail not in response_body:
        raise RuntimeError(f"{path} вернул HTTP {status}: {response_body}")
    if elapsed > _IMMEDIATE_RESPONSE_SECONDS:
        raise RuntimeError(f"{path} ожидал request body дольше {_IMMEDIATE_RESPONSE_SECONDS} seconds")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    arguments = parser.parse_args()

    overlimit_elapsed = _assert_immediate_status(
        arguments.base_url,
        "/api/v1/files/?process=false",
        413,
        "Request body too large",
    )
    denied_elapsed = {
        path: _assert_immediate_status(
            arguments.base_url,
            path,
            403,
            "Resource route disabled for Phase 0",
        )
        for path in _DENIED_RESOURCE_ROUTES
    }
    print(
        json.dumps(
            {
                "host_direct": {"declared_overlimit": "HTTP 413 before body", "elapsed_seconds": overlimit_elapsed},
                "resource_routes": {"status": 403, "elapsed_seconds": denied_elapsed},
            }
        )
    )


if __name__ == "__main__":
    main()
