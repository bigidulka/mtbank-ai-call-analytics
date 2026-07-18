from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from mtbank_ai.pipeline_auth import PipelineBearerAuth


async def _receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def _invoke(headers: object, scope_type: str = "http") -> tuple[list[dict[str, Any]], list[str]]:
    sent: list[dict[str, Any]] = []
    calls: list[str] = []

    async def upstream(
        scope: object,
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        del scope, receive
        calls.append("upstream")
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    app = PipelineBearerAuth(upstream, "a-secure-pipelines-key-with-32-bytes!")
    asyncio.run(app({"type": scope_type, "headers": headers}, _receive, send))
    return sent, calls


def test_rejects_missing_or_wrong_bearer_header_before_upstream() -> None:
    for headers in ([], [(b"authorization", b"Bearer wrong")]):
        sent, calls = _invoke(headers)

        assert calls == []
        assert sent[0]["status"] == 401
        assert sent[1]["body"] == b'{"detail":"Unauthorized"}'


def test_forwards_only_one_exact_bearer_header() -> None:
    sent, calls = _invoke([(b"authorization", b"Bearer a-secure-pipelines-key-with-32-bytes!")])

    assert calls == ["upstream"]
    assert sent[0]["status"] == 204


def test_rejects_ambiguous_duplicate_authorization_headers() -> None:
    sent, calls = _invoke(
        [
            (b"authorization", b"Bearer a-secure-pipelines-key-with-32-bytes!"),
            (b"authorization", b"Bearer a-secure-pipelines-key-with-32-bytes!"),
        ]
    )

    assert calls == []
    assert sent[0]["status"] == 401


def test_preserves_non_http_lifespan_traffic() -> None:
    sent, calls = _invoke([], scope_type="lifespan")

    assert calls == ["upstream"]
    assert sent[0]["status"] == 204
