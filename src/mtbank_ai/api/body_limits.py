"""ASGI body limits that reject requests before parser allocation."""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any
from uuid import UUID, uuid4

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mtbank_ai.domain.errors import DomainError, ErrorCode, build_error_response


class _BodyLimitExceeded(Exception):
    pass


class BodyLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_json_bytes: int,
        max_upload_bytes: int,
        multipart_reserve_bytes: int,
        paths: Collection[str] = ("/analyze", "/analyze/"),
    ) -> None:
        self.app = app
        self._max_json_bytes = max_json_bytes
        self._max_multipart_bytes = max_upload_bytes + multipart_reserve_bytes
        self._paths = frozenset(paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in self._paths:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        limit = self._limit_for_content_type(headers.get("content-type", ""))
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    await self._send_payload_too_large(scope, send)
                    return
            except ValueError:
                pass

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit:
                    raise _BodyLimitExceeded
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyLimitExceeded:
            await self._send_payload_too_large(scope, send)

    def _limit_for_content_type(self, content_type: str) -> int:
        media_type = content_type.partition(";")[0].strip().casefold()
        if media_type == "multipart/form-data":
            return self._max_multipart_bytes
        return self._max_json_bytes

    async def _send_payload_too_large(self, scope: Scope, send: Send) -> None:
        request_id = _request_id(scope)
        _, response = build_error_response(DomainError(ErrorCode.PAYLOAD_TOO_LARGE), request_id)
        body = json.dumps(response.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-request-id", str(request_id).encode()),
        ]
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def _request_id(scope: Scope) -> UUID:
    state = scope.get("state")
    value: Any = state.get("request_id") if isinstance(state, dict) else None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(Headers(scope=scope).get("x-request-id", ""))
    except ValueError:
        return uuid4()
