"""Fail-closed readiness probes for runtime dependencies."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import HttpUrl, SecretStr

from mtbank_ai.application.ports import ReadinessPort
from mtbank_ai.config import SpeechTransportMode


class SpeechHttpReadiness:
    """Проверяет configured speech health без proxy env, redirect или body чтения."""

    def __init__(
        self,
        base_url: HttpUrl,
        timeout_seconds: float,
        *,
        mode: SpeechTransportMode = "internal_http",
        api_key: SecretStr | None = None,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self._health_url = f"{str(base_url).rstrip('/')}/health/ready"
        self._mode = mode
        self._api_key = api_key
        self._client = client_factory(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
            follow_redirects=False,
        )
        self._closed = False

    async def ping(self) -> bool:
        headers = {"Accept-Encoding": "identity"}
        if self._mode == "remote_https":
            if self._api_key is None:
                return False
            headers["Authorization"] = f"Bearer {self._api_key.get_secret_value()}"
        try:
            async with self._client.stream("GET", self._health_url, headers=headers) as response:
                return response.status_code == 200
        except (httpx.HTTPError, TimeoutError):
            return False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()


class CompositeReadiness:
    """Требует readiness каждого explicit runtime dependency."""

    def __init__(self, *dependencies: ReadinessPort) -> None:
        self._dependencies = dependencies
        self._closed = False

    async def ping(self) -> bool:
        results = await asyncio.gather(
            *(dependency.ping() for dependency in self._dependencies),
            return_exceptions=True,
        )
        return all(result is True for result in results)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closed: set[int] = set()
        for dependency in self._dependencies:
            identity = id(dependency)
            if identity in closed:
                continue
            closed.add(identity)
            close = getattr(dependency, "close", None)
            if close is None or not callable(close):
                continue
            result: Any = close()
            if inspect.isawaitable(result):
                await result
