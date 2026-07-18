from __future__ import annotations

import asyncio
import ipaddress
import ssl
from collections.abc import Callable, Coroutine, Iterable

import httpcore
import httpx
import pytest

from mtbank_ai.workflow.fetch import (
    PinnedAddressTransport,
    PinnedTarget,
    SafeUrlFetcher,
    UrlFetchError,
    UrlFetchFailure,
    UrlFetchPolicy,
)


class Resolver:
    def __init__(self, *answers: tuple[str, ...]) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        self.calls.append((hostname, port))
        answer = self._answers.pop(0)
        return tuple(ipaddress.ip_address(value) for value in answer)


class RecordingTransportFactory:
    def __init__(self, handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]) -> None:
        self._handler = handler
        self.targets: list[PinnedTarget] = []

    def __call__(self, target: PinnedTarget) -> httpx.AsyncBaseTransport:
        self.targets.append(target)
        return httpx.MockTransport(self._handler)


class RecordingStream(httpcore.AsyncNetworkStream):
    def __init__(self) -> None:
        self._response_parts = [
            b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: 4\r\nConnection: close\r\n\r\nRIFF",
            b"",
        ]
        self.writes: list[bytes] = []
        self.tls_server_hostnames: list[str | None] = []

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        del max_bytes, timeout
        return self._response_parts.pop(0)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    async def aclose(self) -> None:
        return None

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del ssl_context, timeout
        self.tls_server_hostnames.append(server_hostname)
        return self


class RecordingNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.stream = RecordingStream()
        self.connect_calls: list[tuple[str, int]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del timeout, local_address, socket_options
        self.connect_calls.append((host, port))
        return self.stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del path, timeout, socket_options
        raise AssertionError("Unix socket не должен использоваться")

    async def sleep(self, seconds: float) -> None:
        del seconds


def test_fetcher_accepts_bounded_public_audio_through_one_pinned_resolution() -> None:
    async def scenario() -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"RIFF", request=request)

        resolver = Resolver(("8.8.8.8",))
        transports = RecordingTransportFactory(handler)
        fetcher = SafeUrlFetcher(
            UrlFetchPolicy(max_bytes=32, timeout_seconds=1.0),
            resolver=resolver,
            transport_factory=transports,
        )

        media = await fetcher.fetch("https://media.example.test/call.wav")

        assert media.filename == "call.wav"
        assert media.content_type == "audio/wav"
        assert media.content == b"RIFF"
        assert requests[0].headers["accept"] == "audio/wav, audio/x-wav, audio/mpeg, audio/ogg"
        assert requests[0].headers["host"] == "media.example.test"
        assert resolver.calls == [("media.example.test", 443)]
        assert transports.targets == [
            PinnedTarget(
                hostname="media.example.test",
                port=443,
                address=ipaddress.ip_address("8.8.8.8"),
            )
        ]

    asyncio.run(scenario())


def test_pinned_transport_connects_to_validated_ip_but_preserves_host_sni_and_certificate_name() -> None:
    async def scenario() -> None:
        backend = RecordingNetworkBackend()
        transport = PinnedAddressTransport(
            PinnedTarget(
                hostname="media.example.test",
                port=443,
                address=ipaddress.ip_address("8.8.8.8"),
            ),
            network_backend=backend,
        )
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            response = await client.get(
                "https://media.example.test/call.wav",
                headers={"Authorization": "Bearer must-not-forward", "Cookie": "must-not-forward=true"},
            )
            assert response.content == b"RIFF"

        wire_request = b"".join(backend.stream.writes)
        assert backend.connect_calls == [("8.8.8.8", 443)]
        assert backend.stream.tls_server_hostnames == ["media.example.test"]
        assert b"host: media.example.test\r\n" in wire_request.lower()
        assert b"authorization:" not in wire_request.lower()
        assert b"cookie:" not in wire_request.lower()

    asyncio.run(scenario())


def test_fetcher_blocks_private_target_before_connection() -> None:
    async def scenario() -> None:
        resolver = Resolver(("127.0.0.1",))
        fetcher = SafeUrlFetcher(resolver=resolver)
        with pytest.raises(UrlFetchError) as error:
            await fetcher.fetch("http://internal.example.test/call.wav")
        assert error.value.failure is UrlFetchFailure.BLOCKED_ADDRESS
        assert resolver.calls == [("internal.example.test", 80)]

    asyncio.run(scenario())


def test_fetcher_pins_first_public_answer_without_postrequest_dns_lookup() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"RIFF", request=request)

        resolver = Resolver(("8.8.8.8",), ("127.0.0.1",))
        transports = RecordingTransportFactory(handler)
        fetcher = SafeUrlFetcher(resolver=resolver, transport_factory=transports)

        media = await fetcher.fetch("https://media.example.test/call.wav")

        assert media.content == b"RIFF"
        assert resolver.calls == [("media.example.test", 443)]
        assert transports.targets[0].address == ipaddress.ip_address("8.8.8.8")

    asyncio.run(scenario())


def test_fetcher_reresolves_and_repins_every_redirect_hop() -> None:
    async def scenario() -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "public.example.test":
                return httpx.Response(302, headers={"location": "https://next.example.test/call.wav"}, request=request)
            return httpx.Response(200, headers={"content-type": "audio/wav"}, content=b"RIFF", request=request)

        resolver = Resolver(("8.8.8.8",), ("1.1.1.1",))
        transports = RecordingTransportFactory(handler)
        fetcher = SafeUrlFetcher(resolver=resolver, transport_factory=transports)

        media = await fetcher.fetch("https://public.example.test/call.wav")

        assert media.content == b"RIFF"
        assert [str(request.url) for request in requests] == [
            "https://public.example.test/call.wav",
            "https://next.example.test/call.wav",
        ]
        assert resolver.calls == [("public.example.test", 443), ("next.example.test", 443)]
        assert [(target.hostname, str(target.address)) for target in transports.targets] == [
            ("public.example.test", "8.8.8.8"),
            ("next.example.test", "1.1.1.1"),
        ]

    asyncio.run(scenario())


def test_fetcher_rejects_private_redirect_target_before_requesting_it() -> None:
    async def scenario() -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(302, headers={"location": "http://private.example.test/call.wav"}, request=request)

        resolver = Resolver(("8.8.8.8",), ("10.0.0.1",))
        transports = RecordingTransportFactory(handler)
        fetcher = SafeUrlFetcher(resolver=resolver, transport_factory=transports)
        with pytest.raises(UrlFetchError) as error:
            await fetcher.fetch("https://public.example.test/call.wav")

        assert error.value.failure is UrlFetchFailure.BLOCKED_ADDRESS
        assert [str(request.url) for request in requests] == ["https://public.example.test/call.wav"]
        assert resolver.calls == [("public.example.test", 443), ("private.example.test", 80)]
        assert len(transports.targets) == 1

    asyncio.run(scenario())
