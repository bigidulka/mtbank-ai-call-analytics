"""ASGI pre-body boundary for the pinned OpenWebUI application."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping
from tempfile import SpooledTemporaryFile
from typing import Any
from urllib.parse import unquote_to_bytes

ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[Mapping[str, Any], Receive, Send], Awaitable[None]]

MAX_OPENWEBUI_REQUEST_BODY_BYTES = 25 * 1024 * 1024 + 64 * 1024
_SPOOL_MEMORY_BYTES = 64 * 1024
_REPLAY_CHUNK_BYTES = 64 * 1024
_DENIED_RESOURCE_PATHS = frozenset(
    {
        "/api/v1/retrieval/process/file",
        "/api/v1/retrieval/process/files/batch",
        "/api/v1/retrieval/process/text",
        "/api/v1/retrieval/process/web",
        "/api/v1/retrieval/process/youtube",
        "/api/v1/retrieval/process/web/search",
        "/api/v1/audio/transcriptions",
        "/api/v1/audio/speech",
    }
)
_DENIED_RESOURCE_PATTERNS = (
    re.compile(r"^/api/v1/files/[^/]+/data/content/update$"),
    re.compile(r"^/api/v1/knowledge/[^/]+/file/(?:add|update)$"),
    re.compile(r"^/api/v1/knowledge/[^/]+/files/batch/add$"),
)
_FILE_UPLOAD_PATH = "/api/v1/files"
_CHAT_COMPLETIONS_PATHS = frozenset({"/api/chat/completions", "/api/v1/chat/completions"})
_CHAT_PERSISTENCE_PATH = "/api/v1/chats"
_REMOTE_IMAGE_SCAN_METHODS = frozenset({"POST", "PUT", "PATCH"})


class OpenWebUIPreBodyGuard:
    """Rejects disabled routes and oversized HTTP bodies before OpenWebUI parses them."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = MAX_OPENWEBUI_REQUEST_BODY_BYTES,
        spool_memory_bytes: int = _SPOOL_MEMORY_BYTES,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes должен быть положительным")
        if spool_memory_bytes <= 0:
            raise ValueError("spool_memory_bytes должен быть положительным")
        self._app = app
        self._max_body_bytes = max_body_bytes
        self._spool_memory_bytes = spool_memory_bytes

    async def __call__(self, scope: Mapping[str, Any], receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        scope = _force_file_upload_without_processing(scope)
        if self._is_denied_resource_route(scope):
            await _send_controlled_error(send, 403, "Resource route disabled for Phase 0")
            return

        declared_length = _content_length(scope.get("headers"))
        if declared_length is not None and declared_length > self._max_body_bytes:
            await _send_controlled_error(send, 413, "Request body too large")
            return

        with SpooledTemporaryFile(max_size=self._spool_memory_bytes, mode="w+b") as request_body:
            total = 0
            while True:
                message = await receive()
                message_type = message.get("type")
                if message_type == "http.disconnect":
                    return
                if message_type != "http.request":
                    await _send_controlled_error(send, 400, "Invalid HTTP request body")
                    return

                body = message.get("body", b"")
                if not isinstance(body, bytes):
                    await _send_controlled_error(send, 400, "Invalid HTTP request body")
                    return
                if len(body) > self._max_body_bytes - total:
                    await _send_controlled_error(send, 413, "Request body too large")
                    return

                _write_chunked(request_body, body)
                total += len(body)
                if not message.get("more_body", False):
                    break

            request_body.seek(0)
            if _contains_remote_image_url_in_request(scope, request_body):
                await _send_controlled_error(send, 403, "Remote image URLs disabled for Phase 0")
                return

            request_body.seek(0)
            await self._app(scope, _ReplayReceive(request_body, total, receive), send)

    @staticmethod
    def _is_denied_resource_route(scope: Mapping[str, Any]) -> bool:
        method = scope.get("method")
        path = scope.get("path")
        if not isinstance(method, str) or method.upper() != "POST" or not isinstance(path, str):
            return False

        normalized_path = path.rstrip("/")
        return normalized_path in _DENIED_RESOURCE_PATHS or any(
            pattern.fullmatch(normalized_path) for pattern in _DENIED_RESOURCE_PATTERNS
        )


class _ReplayReceive:
    """Replays a bounded temporary request body, then resumes the client receive channel."""

    def __init__(self, request_body: Any, total: int, receive: Receive) -> None:
        self._request_body = request_body
        self._remaining = total
        self._receive = receive
        self._must_send_empty_request = total == 0

    async def __call__(self) -> ASGIMessage:
        if self._remaining:
            chunk = self._request_body.read(min(_REPLAY_CHUNK_BYTES, self._remaining))
            self._remaining -= len(chunk)
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": self._remaining > 0,
            }
        if self._must_send_empty_request:
            self._must_send_empty_request = False
            return {"type": "http.request", "body": b"", "more_body": False}
        return await self._receive()


def _force_file_upload_without_processing(scope: Mapping[str, Any]) -> Mapping[str, Any]:
    method = scope.get("method")
    path = scope.get("path")
    if (
        not isinstance(method, str)
        or method.upper() != "POST"
        or not isinstance(path, str)
        or path.rstrip("/") != _FILE_UPLOAD_PATH
    ):
        return scope

    query_string = scope.get("query_string")
    parameters = query_string.split(b"&") if isinstance(query_string, bytes) and query_string else []
    parameters = [parameter for parameter in parameters if _query_parameter_name(parameter) != b"process"]
    parameters.append(b"process=false")

    upstream_scope = dict(scope)
    upstream_scope["query_string"] = b"&".join(parameters)
    return upstream_scope


def _query_parameter_name(parameter: bytes) -> bytes:
    raw_name = parameter.partition(b"=")[0].replace(b"+", b" ")
    return unquote_to_bytes(raw_name)


def _contains_remote_image_url_in_request(scope: Mapping[str, Any], request_body: Any) -> bool:
    method = scope.get("method")
    path = scope.get("path")
    if not isinstance(method, str) or method.upper() not in _REMOTE_IMAGE_SCAN_METHODS or not isinstance(path, str):
        return False

    normalized_path = path.rstrip("/")
    is_completion = normalized_path in _CHAT_COMPLETIONS_PATHS
    is_chat_persistence = normalized_path == _CHAT_PERSISTENCE_PATH or normalized_path.startswith(
        f"{_CHAT_PERSISTENCE_PATH}/"
    )
    if not is_completion and not is_chat_persistence:
        return False

    try:
        payload = json.load(request_body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return False

    pending: list[object] = [payload]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if _is_remote_image_url(item.get("image_url")):
                return True
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return False


def _is_remote_image_url(value: object) -> bool:
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str):
        return False

    scheme, separator, _ = value.strip().partition(":")
    return bool(separator) and scheme.casefold() in {"http", "https"}


def _content_length(headers: object) -> int | None:
    if not isinstance(headers, list):
        return None

    values: list[bytes] = []
    for header in headers:
        if not isinstance(header, tuple) or len(header) != 2:
            return None
        name, value = header
        if not isinstance(name, bytes) or not isinstance(value, bytes):
            return None
        if name.lower() == b"content-length":
            values.append(value)

    if len(values) != 1 or not values[0].isdigit():
        return None
    return int(values[0])


def _write_chunked(request_body: Any, body: bytes) -> None:
    for offset in range(0, len(body), _REPLAY_CHUNK_BYTES):
        request_body.write(body[offset : offset + _REPLAY_CHUNK_BYTES])


async def _send_controlled_error(send: Send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
