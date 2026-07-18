"""Fail-closed URL media fetcher с DNS-pinned transport и redirect revalidation."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, NoReturn, Protocol, cast
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpcore
import httpx
from pydantic import Field, field_validator, model_validator

from mtbank_ai.domain.base import MimeType, PositiveFloat, PositiveInt, StrictFrozenModel


class UrlFetchFailure(StrEnum):
    INVALID_URL = "invalid_url"
    BLOCKED_ADDRESS = "blocked_address"
    REDIRECT_LIMIT = "redirect_limit"
    UNSUPPORTED_MEDIA = "unsupported_media"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class UrlFetchError(RuntimeError):
    """Sanitized URL fetch failure без URL, DNS или upstream response details."""

    def __init__(self, failure: UrlFetchFailure) -> None:
        self.failure = failure
        super().__init__(failure.value)


class UrlFetchPolicy(StrictFrozenModel):
    max_bytes: PositiveInt = 25 * 1024 * 1024
    timeout_seconds: PositiveFloat = 15.0
    max_redirects: Annotated[int, Field(ge=0, le=5)] = 3
    allowed_media_types: tuple[MimeType, ...] = (
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/ogg",
    )

    @field_validator("allowed_media_types", mode="before")
    @classmethod
    def parse_media_types(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_media_types(self) -> UrlFetchPolicy:
        if not self.allowed_media_types or len(set(self.allowed_media_types)) != len(self.allowed_media_types):
            raise ValueError("allowed_media_types должны быть непустыми и уникальными")
        return self


@dataclass(frozen=True, slots=True)
class FetchedUrlMedia:
    filename: str
    content_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class PinnedTarget:
    """Один проверенный DNS target, неизменно используемый для одного request hop."""

    hostname: str
    port: int
    address: ipaddress.IPv4Address | ipaddress.IPv6Address


class HostResolver(Protocol):
    async def resolve(self, hostname: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]: ...


class PinnedTransportFactory(Protocol):
    def __call__(self, target: PinnedTarget) -> httpx.AsyncBaseTransport: ...


class SystemHostResolver:
    async def resolve(self, hostname: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        try:
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except OSError as error:
            raise UrlFetchError(UrlFetchFailure.UNAVAILABLE) from error
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for info in infos:
            try:
                address = ipaddress.ip_address(info[4][0])
            except ValueError as error:
                raise UrlFetchError(UrlFetchFailure.BLOCKED_ADDRESS) from error
            if address not in addresses:
                addresses.append(address)
        if not addresses:
            raise UrlFetchError(UrlFetchFailure.UNAVAILABLE)
        return tuple(addresses)


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Подменяет только TCP target, не меняя HTTP authority или TLS hostname."""

    def __init__(
        self,
        target: PinnedTarget,
        *,
        delegate: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._target = target
        self._delegate: httpcore.AsyncNetworkBackend = delegate or cast(
            httpcore.AsyncNetworkBackend,
            httpcore.AnyIOBackend(),
        )

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if host != self._target.hostname or port != self._target.port:
            raise httpcore.ConnectError("pinned transport received an unexpected origin")
        return await self._delegate.connect_tcp(
            str(self._target.address),
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del path, timeout, socket_options
        raise httpcore.ConnectError("pinned transport does not permit Unix sockets")

    async def sleep(self, seconds: float) -> None:
        await self._delegate.sleep(seconds)


class _HttpcoreResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterable[bytes]) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for part in self._stream:
                yield part
        except _HTTPCORE_ERRORS as error:
            _raise_httpx_error(error)

    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except _HTTPCORE_ERRORS as error:
                _raise_httpx_error(error)


class PinnedAddressTransport(httpx.AsyncBaseTransport):
    """HTTPX transport, который соединяется только с заранее проверенным IP address."""

    def __init__(
        self,
        target: PinnedTarget,
        *,
        network_backend: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._target = target
        self._pool = httpcore.AsyncConnectionPool(
            max_connections=1,
            max_keepalive_connections=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=_PinnedNetworkBackend(target, delegate=network_backend),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
        if request.url.host != self._target.hostname or request_port != self._target.port:
            raise httpx.UnsupportedProtocol("pinned transport received an unexpected origin")
        headers = _pinned_headers(request)
        extensions = dict(request.extensions)
        # httpcore передаёт это значение одновременно в TLS SNI и certificate verification.
        extensions["sni_hostname"] = self._target.hostname
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=headers,
            content=request.stream,
            extensions=extensions,
        )
        try:
            response = await self._pool.handle_async_request(core_request)
        except _HTTPCORE_ERRORS as error:
            _raise_httpx_error(error)
        assert isinstance(response.stream, AsyncIterable)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_HttpcoreResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


def _pinned_headers(request: httpx.Request) -> list[tuple[bytes, bytes]]:
    blocked = {b"authorization", b"cookie", b"proxy-authorization", b"host"}
    headers = [(name, value) for name, value in request.headers.raw if name.lower() not in blocked]
    headers.append((b"host", request.url.netloc))
    return headers


_HTTPCORE_ERRORS = (
    httpcore.ConnectTimeout,
    httpcore.ReadTimeout,
    httpcore.WriteTimeout,
    httpcore.PoolTimeout,
    httpcore.TimeoutException,
    httpcore.ConnectError,
    httpcore.ReadError,
    httpcore.WriteError,
    httpcore.NetworkError,
    httpcore.ProxyError,
    httpcore.RemoteProtocolError,
    httpcore.LocalProtocolError,
    httpcore.ProtocolError,
    httpcore.UnsupportedProtocol,
)


def _raise_httpx_error(error: Exception) -> NoReturn:
    error_types: tuple[tuple[type[Exception], type[httpx.HTTPError]], ...] = (
        (httpcore.ConnectTimeout, httpx.ConnectTimeout),
        (httpcore.ReadTimeout, httpx.ReadTimeout),
        (httpcore.WriteTimeout, httpx.WriteTimeout),
        (httpcore.PoolTimeout, httpx.PoolTimeout),
        (httpcore.TimeoutException, httpx.TimeoutException),
        (httpcore.ConnectError, httpx.ConnectError),
        (httpcore.ReadError, httpx.ReadError),
        (httpcore.WriteError, httpx.WriteError),
        (httpcore.NetworkError, httpx.NetworkError),
        (httpcore.ProxyError, httpx.ProxyError),
        (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
        (httpcore.LocalProtocolError, httpx.LocalProtocolError),
        (httpcore.ProtocolError, httpx.ProtocolError),
        (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    )
    for source_type, target_type in error_types:
        if isinstance(error, source_type):
            raise target_type(str(error)) from error
    raise error


class SafeUrlFetcher:
    """Скачивает только bounded public audio через no-proxy DNS-pinned transport."""

    def __init__(
        self,
        policy: UrlFetchPolicy | None = None,
        *,
        resolver: HostResolver | None = None,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
        transport_factory: PinnedTransportFactory = PinnedAddressTransport,
    ) -> None:
        self._policy = policy or UrlFetchPolicy()
        self._resolver = resolver or SystemHostResolver()
        self._client_factory = client_factory
        self._transport_factory = transport_factory

    async def fetch(self, source_url: str) -> FetchedUrlMedia:
        current_url = _normalize_url(source_url)
        try:
            async with asyncio.timeout(self._policy.timeout_seconds):
                for redirect_count in range(self._policy.max_redirects + 1):
                    target = await self._resolve_pinned_target(current_url)
                    async with self._client_factory(
                        follow_redirects=False,
                        trust_env=False,
                        timeout=httpx.Timeout(self._policy.timeout_seconds),
                        transport=self._transport_factory(target),
                    ) as client:
                        response = await self._request(client, current_url)
                        try:
                            location = response.headers.get("location")
                            if response.status_code in {301, 302, 303, 307, 308}:
                                if redirect_count >= self._policy.max_redirects or not location:
                                    raise UrlFetchError(UrlFetchFailure.REDIRECT_LIMIT)
                                current_url = _normalize_url(urljoin(current_url, location))
                                continue
                            if not 200 <= response.status_code < 300:
                                raise UrlFetchError(UrlFetchFailure.UNAVAILABLE)
                            content_type = _content_type(response.headers.get("content-type"))
                            if content_type not in self._policy.allowed_media_types:
                                raise UrlFetchError(UrlFetchFailure.UNSUPPORTED_MEDIA)
                            _validate_declared_size(response.headers.get("content-length"), self._policy.max_bytes)
                            content = await _read_bounded(response, self._policy.max_bytes)
                            if not content:
                                raise UrlFetchError(UrlFetchFailure.UNSUPPORTED_MEDIA)
                            return FetchedUrlMedia(
                                filename=_safe_filename(current_url),
                                content_type=content_type,
                                content=content,
                            )
                        finally:
                            await response.aclose()
        except UrlFetchError:
            raise
        except TimeoutError:
            raise UrlFetchError(UrlFetchFailure.TIMEOUT) from None
        except httpx.HTTPError:
            raise UrlFetchError(UrlFetchFailure.UNAVAILABLE) from None
        except Exception:
            raise UrlFetchError(UrlFetchFailure.UNAVAILABLE) from None
        raise UrlFetchError(UrlFetchFailure.REDIRECT_LIMIT)

    async def _request(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        try:
            return await client.send(
                client.build_request(
                    "GET",
                    url,
                    headers={
                        "Accept": ", ".join(self._policy.allowed_media_types),
                        "Host": _host_header(url),
                        "User-Agent": "mtbank-ai-media-fetch/1",
                    },
                ),
                stream=True,
            )
        except httpx.HTTPError:
            raise UrlFetchError(UrlFetchFailure.UNAVAILABLE) from None

    async def _resolve_pinned_target(self, url: str) -> PinnedTarget:
        parts = urlsplit(url)
        assert parts.hostname is not None
        if _is_metadata_hostname(parts.hostname):
            raise UrlFetchError(UrlFetchFailure.BLOCKED_ADDRESS)
        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = await self._resolver.resolve(parts.hostname, port)
        if any(not _is_public_address(address) for address in addresses):
            raise UrlFetchError(UrlFetchFailure.BLOCKED_ADDRESS)
        return PinnedTarget(hostname=parts.hostname, port=port, address=addresses[0])


def _normalize_url(value: object) -> str:
    if not isinstance(value, str) or len(value) > 4_096 or value != value.strip():
        raise UrlFetchError(UrlFetchFailure.INVALID_URL)
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as error:
        raise UrlFetchError(UrlFetchFailure.INVALID_URL) from error
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
        or port == 0
    ):
        raise UrlFetchError(UrlFetchFailure.INVALID_URL)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def _content_type(value: object) -> str:
    if not isinstance(value, str):
        raise UrlFetchError(UrlFetchFailure.UNSUPPORTED_MEDIA)
    content_type = value.partition(";")[0].strip().casefold()
    if not content_type:
        raise UrlFetchError(UrlFetchFailure.UNSUPPORTED_MEDIA)
    return content_type


def _validate_declared_size(value: object, max_bytes: int) -> None:
    if value is None:
        return
    try:
        size = int(cast(str, value))
    except (TypeError, ValueError):
        raise UrlFetchError(UrlFetchFailure.PAYLOAD_TOO_LARGE) from None
    if size < 0 or size > max_bytes:
        raise UrlFetchError(UrlFetchFailure.PAYLOAD_TOO_LARGE)


async def _read_bounded(response: httpx.Response, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise UrlFetchError(UrlFetchFailure.PAYLOAD_TOO_LARGE)
        chunks.append(chunk)
    return b"".join(chunks)


def _is_metadata_hostname(hostname: str) -> bool:
    normalized = hostname.casefold().rstrip(".")
    return (
        normalized
        in {
            "localhost",
            "localhost.localdomain",
            "metadata",
            "metadata.google.internal",
            "instance-data",
        }
        or normalized.endswith(".localhost")
        or normalized.endswith(".local")
    )


def _is_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _host_header(url: str) -> str:
    parts = urlsplit(url)
    assert parts.hostname is not None
    hostname = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    if parts.port is not None:
        return f"{hostname}:{parts.port}"
    return hostname


def _safe_filename(url: str) -> str:
    path = unquote(urlsplit(url).path)
    candidate = PurePosixPath(path).name.strip()
    if not candidate or len(candidate) > 128 or any(ord(character) < 32 for character in candidate):
        return "audio"
    return candidate
