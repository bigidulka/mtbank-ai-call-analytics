from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from starlette.responses import StreamingResponse

from mtbank_ai.openwebui_guard import MAX_OPENWEBUI_REQUEST_BODY_BYTES, OpenWebUIPreBodyGuard


def _request_messages(total: int, *, chunk_size: int = 64 * 1024) -> Iterator[dict[str, Any]]:
    remaining = total
    while remaining:
        size = min(remaining, chunk_size)
        remaining -= size
        yield {"type": "http.request", "body": b"x" * size, "more_body": remaining > 0}


def _invoke(
    scope: dict[str, Any],
    messages: Iterator[dict[str, Any]],
    *,
    max_body_bytes: int = MAX_OPENWEBUI_REQUEST_BODY_BYTES,
) -> tuple[list[dict[str, Any]], list[str], bytearray, int, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    calls: list[str] = []
    replayed = bytearray()
    receive_calls = 0
    upstream_scopes: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        return next(messages)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def upstream(
        upstream_scope: Any,
        upstream_receive: Any,
        upstream_send: Any,
    ) -> None:
        upstream_scopes.append(dict(upstream_scope))
        calls.append("upstream")
        while True:
            message = await upstream_receive()
            replayed.extend(message["body"])
            if not message["more_body"]:
                break
        await upstream_send({"type": "http.response.start", "status": 201, "headers": []})
        await upstream_send({"type": "http.response.body", "body": b"first", "more_body": True})
        await upstream_send({"type": "http.response.body", "body": b"second", "more_body": False})

    app = OpenWebUIPreBodyGuard(upstream, max_body_bytes=max_body_bytes, spool_memory_bytes=4)
    asyncio.run(app(scope, receive, send))
    return sent, calls, replayed, receive_calls, upstream_scopes


def _http_scope(
    path: str = "/api/v1/files/",
    headers: list[tuple[bytes, bytes]] | None = None,
    *,
    method: str = "POST",
    query_string: bytes = b"",
) -> dict[str, Any]:
    return {"type": "http", "method": method, "path": path, "headers": headers or [], "query_string": query_string}


def test_default_cap_matches_attachment_limit_plus_multipart_reserve() -> None:
    assert MAX_OPENWEBUI_REQUEST_BODY_BYTES == 25 * 1024 * 1024 + 64 * 1024


def test_allows_exact_content_length_and_replays_bounded_body() -> None:
    exact_size = MAX_OPENWEBUI_REQUEST_BODY_BYTES
    sent, calls, replayed, _, _ = _invoke(
        _http_scope(headers=[(b"content-length", str(exact_size).encode("ascii"))]),
        _request_messages(exact_size),
    )

    assert calls == ["upstream"]
    assert len(replayed) == exact_size
    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
        "http.response.body",
    ]
    assert sent[1]["more_body"] is True
    assert sent[2]["body"] == b"second"


def test_rejects_over_content_length_without_reading_body() -> None:
    def no_receive() -> Iterator[dict[str, Any]]:
        raise AssertionError("guard не должен читать body при overlimit Content-Length")
        yield {}

    sent, calls, replayed, receive_calls, _ = _invoke(
        _http_scope(headers=[(b"content-length", str(MAX_OPENWEBUI_REQUEST_BODY_BYTES + 1).encode("ascii"))]),
        no_receive(),
    )

    assert calls == []
    assert replayed == b""
    assert receive_calls == 0
    assert sent[0]["status"] == 413
    assert sent[1]["body"] == b'{"detail":"Request body too large"}'


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]], ids=["chunked", "lying-content-length"])
def test_rejects_chunked_or_lying_overlimit_before_upstream(headers: list[tuple[bytes, bytes]]) -> None:
    sent, calls, replayed, _, _ = _invoke(
        _http_scope(headers=headers),
        iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"56789", "more_body": False},
            ]
        ),
        max_body_bytes=8,
    )

    assert calls == []
    assert replayed == b""
    assert sent[0]["status"] == 413


def test_replays_under_limit_without_buffering_response_stream() -> None:
    sent, calls, replayed, _, _ = _invoke(
        _http_scope(),
        iter(
            [
                {"type": "http.request", "body": b"audio-", "more_body": True},
                {"type": "http.request", "body": b"probe", "more_body": False},
            ]
        ),
        max_body_bytes=16,
    )

    assert calls == ["upstream"]
    assert replayed == b"audio-probe"
    assert sent[1] == {"type": "http.response.body", "body": b"first", "more_body": True}
    assert sent[2] == {"type": "http.response.body", "body": b"second", "more_body": False}


def test_waits_for_real_disconnect_while_starlette_streams_asgi_23_response() -> None:
    sent: list[dict[str, Any]] = []
    receive_calls = 0

    async def invoke() -> None:
        nonlocal receive_calls
        request_sent = False
        disconnected = asyncio.Event()

        async def receive() -> dict[str, Any]:
            nonlocal request_sent, receive_calls
            receive_calls += 1
            if not request_sent:
                request_sent = True
                return {"type": "http.request", "body": b"buffered-request", "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        async def delayed_chunks() -> AsyncIterator[bytes]:
            yield b"first"
            await asyncio.sleep(0.01)
            yield b"second"

        response = StreamingResponse(delayed_chunks(), media_type="text/event-stream")

        async def upstream(upstream_scope: Any, upstream_receive: Any, upstream_send: Any) -> None:
            await response(upstream_scope, upstream_receive, upstream_send)

        scope = _http_scope("/stream")
        scope["asgi"] = {"version": "3.0", "spec_version": "2.3"}
        app = OpenWebUIPreBodyGuard(upstream)
        await asyncio.wait_for(app(scope, receive, send), timeout=1)

    asyncio.run(invoke())

    assert receive_calls == 2
    assert [message.get("body", b"") for message in sent if message["type"] == "http.response.body"] == [
        b"first",
        b"second",
        b"",
    ]


@pytest.mark.parametrize(
    ("path", "query_string", "expected_query_string"),
    [
        ("/api/v1/files/", b"", b"process=false"),
        ("/api/v1/files", b"process=true", b"process=false"),
        ("/api/v1/files/", b"process=true&process=false", b"process=false"),
        ("/api/v1/files/", b"process=false&process=true", b"process=false"),
        ("/api/v1/files/", b"process=maybe&process&keep=1", b"keep=1&process=false"),
        ("/api/v1/files/", b"keep=%2F&pr%6fcess=true&tag=a%20b", b"keep=%2F&tag=a%20b&process=false"),
    ],
)
def test_forces_file_upload_processing_off_without_changing_body(
    path: str,
    query_string: bytes,
    expected_query_string: bytes,
) -> None:
    sent, calls, replayed, _, upstream_scopes = _invoke(
        _http_scope(path, query_string=query_string),
        iter([{"type": "http.request", "body": b"audio-probe", "more_body": False}]),
        max_body_bytes=16,
    )

    assert calls == ["upstream"]
    assert replayed == b"audio-probe"
    assert upstream_scopes[0]["query_string"] == expected_query_string
    assert sent[0]["status"] == 201


@pytest.mark.parametrize("path", ["/api/chat/completions", "/api/v1/chat/completions"])
@pytest.mark.parametrize(
    "image_url",
    [
        {"url": "https://example.invalid/unbounded-image"},
        " HTTP://example.invalid/unbounded-image ",
    ],
    ids=["object", "string-uppercase"],
)
def test_rejects_remote_structured_image_url_before_upstream(path: str, image_url: object) -> None:
    body = json.dumps(
        {
            "model": "pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Проверь вложение."},
                        {"type": "image_url", "image_url": image_url},
                    ],
                }
            ],
            "files": [{"type": "file", "id": "attachment-id"}],
        }
    ).encode("utf-8")

    sent, calls, replayed, receive_calls, _ = _invoke(
        _http_scope(
            path,
            headers=[
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        ),
        iter([{"type": "http.request", "body": body, "more_body": False}]),
        max_body_bytes=len(body) + 1,
    )

    assert calls == []
    assert replayed == b""
    assert receive_calls == 1
    assert sent[0]["status"] == 403
    assert sent[1]["body"] == b'{"detail":"Remote image URLs disabled for Phase 0"}'


def test_rejects_remote_image_url_in_user_message_on_v1_alias() -> None:
    body = json.dumps(
        {
            "model": "pipeline",
            "messages": [{"role": "user", "content": "Безопасный client message."}],
            "user_message": {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/persisted-image"},
                    }
                ],
            },
        }
    ).encode("utf-8")

    sent, calls, replayed, _, _ = _invoke(
        _http_scope("/api/v1/chat/completions", headers=[(b"content-type", b"application/json")]),
        iter([{"type": "http.request", "body": body, "more_body": False}]),
        max_body_bytes=len(body) + 1,
    )

    assert calls == []
    assert replayed == b""
    assert sent[0]["status"] == 403


@pytest.mark.parametrize("path", ["/api/v1/chats/new", "/api/v1/chats/chat-id"])
def test_rejects_remote_image_url_in_chat_persistence_writes(path: str) -> None:
    body = json.dumps(
        {
            "chat": {
                "history": {
                    "messages": {
                        "message-id": {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "http://example.invalid/persisted-image"},
                                }
                            ],
                        }
                    }
                }
            }
        }
    ).encode("utf-8")

    sent, calls, replayed, _, _ = _invoke(
        _http_scope(path, headers=[(b"content-type", b"application/json")]),
        iter([{"type": "http.request", "body": body, "more_body": False}]),
        max_body_bytes=len(body) + 1,
    )

    assert calls == []
    assert replayed == b""
    assert sent[0]["status"] == 403


@pytest.mark.parametrize("path", ["/api/chat/completions", "/api/v1/chat/completions"])
def test_keeps_text_and_top_level_files_chat_body_unchanged(path: str) -> None:
    body = json.dumps(
        {
            "model": "pipeline",
            "messages": [{"role": "user", "content": "Проверь приложенный WAV."}],
            "files": [
                {
                    "type": "file",
                    "id": "attachment-id",
                    "file": {"id": "attachment-id", "filename": "sample.wav"},
                }
            ],
            "stream": False,
        }
    ).encode("utf-8")

    sent, calls, replayed, _, upstream_scopes = _invoke(
        _http_scope(path, headers=[(b"content-type", b"application/json")]),
        iter([{"type": "http.request", "body": body, "more_body": False}]),
        max_body_bytes=len(body) + 1,
    )

    assert calls == ["upstream"]
    assert replayed == body
    assert upstream_scopes[0]["path"] == path
    assert sent[0]["status"] == 201


def test_keeps_safe_chat_persistence_write_unchanged() -> None:
    body = json.dumps(
        {
            "chat": {
                "title": "Phase 0",
                "history": {
                    "messages": {
                        "message-id": {"role": "user", "content": "Проверь приложенный WAV."},
                    }
                },
            }
        }
    ).encode("utf-8")

    sent, calls, replayed, _, upstream_scopes = _invoke(
        _http_scope("/api/v1/chats/new", headers=[(b"content-type", b"application/json")]),
        iter([{"type": "http.request", "body": body, "more_body": False}]),
        max_body_bytes=len(body) + 1,
    )

    assert calls == ["upstream"]
    assert replayed == body
    assert upstream_scopes[0]["path"] == "/api/v1/chats/new"
    assert sent[0]["status"] == 201


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/retrieval/process/file",
        "/api/v1/retrieval/process/files/batch/",
        "/api/v1/retrieval/process/text",
        "/api/v1/files/attachment-id/data/content/update",
        "/api/v1/knowledge/knowledge-id/file/add",
        "/api/v1/knowledge/knowledge-id/file/update",
        "/api/v1/knowledge/knowledge-id/files/batch/add",
        "/api/v1/retrieval/process/web",
        "/api/v1/retrieval/process/youtube/",
        "/api/v1/retrieval/process/web/search",
        "/api/v1/audio/transcriptions",
        "/api/v1/audio/speech",
    ],
)
def test_denies_resource_routes_before_body_or_upstream(path: str) -> None:
    def no_receive() -> Iterator[dict[str, Any]]:
        raise AssertionError("denied route не должен читать body")
        yield {}

    sent, calls, replayed, receive_calls, _ = _invoke(
        _http_scope(path, headers=[(b"content-length", b"999999999")]),
        no_receive(),
    )

    assert calls == []
    assert replayed == b""
    assert receive_calls == 0
    assert sent[0]["status"] == 403
    assert sent[1]["body"] == b'{"detail":"Resource route disabled for Phase 0"}'


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/files/attachment-id",
        "/api/v1/files/attachment-id/process/status",
        "/api/v1/files/attachment-id/data/content",
        "/api/v1/files/attachment-id/content",
    ],
)
def test_keeps_file_metadata_and_content_fetch_routes_available(path: str) -> None:
    sent, calls, replayed, _, upstream_scopes = _invoke(
        _http_scope(path, method="GET", query_string=b"download=true"),
        iter([{"type": "http.request", "body": b"", "more_body": False}]),
    )

    assert calls == ["upstream"]
    assert replayed == b""
    assert upstream_scopes[0]["query_string"] == b"download=true"
    assert sent[0]["status"] == 201


@pytest.mark.parametrize("scope_type", ["websocket", "lifespan"])
def test_passes_non_http_scopes_through_unchanged(scope_type: str) -> None:
    sent: list[dict[str, Any]] = []
    calls: list[tuple[object, object]] = []

    async def receive() -> dict[str, Any]:
        return {"type": f"{scope_type}.noop"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def upstream(scope: object, upstream_receive: object, upstream_send: Any) -> None:
        calls.append((scope, upstream_receive))
        await upstream_send({"type": "passthrough"})

    scope = {"type": scope_type}
    receive_callable = receive
    app = OpenWebUIPreBodyGuard(upstream)
    asyncio.run(app(scope, receive_callable, send))

    assert calls == [(scope, receive_callable)]
    assert sent == [{"type": "passthrough"}]
