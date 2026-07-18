from __future__ import annotations

import asyncio

import httpx
from pydantic import HttpUrl, SecretStr, TypeAdapter

from mtbank_ai.api.readiness import CompositeReadiness, SpeechHttpReadiness


def test_speech_readiness_uses_bounded_direct_http_client_without_reading_body() -> None:
    async def scenario() -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL("http://speech:8010/health/ready")
            assert request.headers["accept-encoding"] == "identity"
            assert "authorization" not in request.headers
            return httpx.Response(200, content=b'{"status":"ready"}')

        transport = httpx.MockTransport(handler)

        def client_factory(
            *, timeout: httpx.Timeout, trust_env: bool, follow_redirects: bool
        ) -> httpx.AsyncClient:
            captured.update(
                timeout=timeout,
                trust_env=trust_env,
                follow_redirects=follow_redirects,
            )
            return httpx.AsyncClient(
                transport=transport,
                timeout=timeout,
                trust_env=trust_env,
                follow_redirects=follow_redirects,
            )

        readiness = SpeechHttpReadiness(
            TypeAdapter(HttpUrl).validate_python("http://speech:8010"),
            1.5,
            client_factory=client_factory,
        )
        assert await readiness.ping()
        await readiness.close()
        await readiness.close()

        assert captured["trust_env"] is False
        assert captured["follow_redirects"] is False
        assert captured["timeout"] == httpx.Timeout(1.5)

    asyncio.run(scenario())


def test_speech_readiness_fails_closed_for_non_ready_or_transport_failure() -> None:
    async def scenario() -> None:
        def unavailable_handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(503)

        def unavailable_client_factory(
            *, timeout: httpx.Timeout, trust_env: bool, follow_redirects: bool
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.MockTransport(unavailable_handler),
                timeout=timeout,
                trust_env=trust_env,
                follow_redirects=follow_redirects,
            )

        unavailable = SpeechHttpReadiness(
            TypeAdapter(HttpUrl).validate_python("http://speech:8010"),
            1.0,
            client_factory=unavailable_client_factory,
        )
        assert not await unavailable.ping()
        await unavailable.close()

        def failed_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("unavailable", request=request)

        def failed_client_factory(
            *, timeout: httpx.Timeout, trust_env: bool, follow_redirects: bool
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.MockTransport(failed_handler),
                timeout=timeout,
                trust_env=trust_env,
                follow_redirects=follow_redirects,
            )

        failed = SpeechHttpReadiness(
            TypeAdapter(HttpUrl).validate_python("http://speech:8010"),
            1.0,
            client_factory=failed_client_factory,
        )
        assert not await failed.ping()
        await failed.close()

    asyncio.run(scenario())


def test_speech_readiness_uses_bearer_only_for_remote_https() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url == httpx.URL("https://speech.example.test/health/ready")
            assert request.headers["authorization"] == "Bearer opaque-remote-speech-key"
            return httpx.Response(200)

        def client_factory(
            *, timeout: httpx.Timeout, trust_env: bool, follow_redirects: bool
        ) -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
                timeout=timeout,
                trust_env=trust_env,
                follow_redirects=follow_redirects,
            )

        readiness = SpeechHttpReadiness(
            TypeAdapter(HttpUrl).validate_python("https://speech.example.test"),
            1.0,
            mode="remote_https",
            api_key=SecretStr("opaque-remote-speech-key"),
            client_factory=client_factory,
        )
        assert await readiness.ping()
        await readiness.close()

        missing_key = SpeechHttpReadiness(
            TypeAdapter(HttpUrl).validate_python("https://speech.example.test"),
            1.0,
            mode="remote_https",
            client_factory=client_factory,
        )
        assert not await missing_key.ping()
        await missing_key.close()

    asyncio.run(scenario())


def test_composite_readiness_runs_dependencies_concurrently_and_closes_once() -> None:
    class Dependency:
        def __init__(self, result: bool) -> None:
            self.result = result
            self.pings = 0
            self.closes = 0

        async def ping(self) -> bool:
            self.pings += 1
            return self.result

        async def close(self) -> None:
            self.closes += 1

    async def scenario() -> None:
        class BlockingDependency(Dependency):
            def __init__(self, result: bool) -> None:
                super().__init__(result)
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def ping(self) -> bool:
                self.pings += 1
                self.started.set()
                await self.release.wait()
                return self.result

        database = BlockingDependency(False)
        speech = BlockingDependency(True)
        readiness = CompositeReadiness(database, speech, database)
        ping = asyncio.create_task(readiness.ping())

        await asyncio.wait_for(asyncio.gather(database.started.wait(), speech.started.wait()), timeout=0.1)
        database.release.set()
        speech.release.set()
        assert not await ping
        assert database.pings == 2
        assert speech.pings == 1
        await readiness.close()
        await readiness.close()
        assert database.closes == 1
        assert speech.closes == 1

    asyncio.run(scenario())
