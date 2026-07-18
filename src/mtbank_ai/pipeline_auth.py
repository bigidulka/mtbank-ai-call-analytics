"""ASGI-граница аутентификации для всех HTTP маршрутов Pipelines."""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[Mapping[str, Any], Receive, Send], Awaitable[None]]


class PipelineBearerAuth:
    """Требует точный Bearer key до передачи HTTP-запроса upstream приложению."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self._app = app
        self._expected_authorization = f"Bearer {api_key}".encode()

    async def __call__(
        self,
        scope: Mapping[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        if self._is_authorized(scope.get("headers")):
            await self._app(scope, receive, send)
            return

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", b"25"),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b'{"detail":"Unauthorized"}'})

    def _is_authorized(self, headers: object) -> bool:
        if not isinstance(headers, list):
            return False

        values: list[bytes] = []
        for header in headers:
            if not isinstance(header, tuple) or len(header) != 2:
                return False
            name, value = header
            if not isinstance(name, bytes) or not isinstance(value, bytes):
                return False
            if name.lower() == b"authorization":
                values.append(value)

        return len(values) == 1 and hmac.compare_digest(values[0], self._expected_authorization)
