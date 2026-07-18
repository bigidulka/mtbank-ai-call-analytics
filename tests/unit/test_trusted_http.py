from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request

import pytest

import pipeline
from mtbank_ai.trusted_http import TrustedHttpError, build_trusted_opener, require_exact_base_url
from pipeline import FileFetchError, OpenWebUIFileClient


class _RecordingServer:
    def __init__(self) -> None:
        self.records: list[tuple[str, str | None, bytes]] = []
        self.location: str | None = None
        self.status = 200
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.parent = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        address = self._server.server_address
        return f"http://{address[0]}:{address[1]}"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    @staticmethod
    def _handler() -> type[BaseHTTPRequestHandler]:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self._respond()

            def do_POST(self) -> None:  # noqa: N802
                self._respond()

            def _respond(self) -> None:
                parent = self.server.parent  # type: ignore[attr-defined]
                length = int(self.headers.get("Content-Length", "0"))
                parent.records.append((self.path, self.headers.get("Authorization"), self.rfile.read(length)))
                self.send_response(parent.status)
                if parent.location is not None:
                    self.send_header("Location", parent.location)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        return Handler


@contextmanager
def _recording_server() -> Iterator[_RecordingServer]:
    server = _RecordingServer()
    server.start()
    try:
        yield server
    finally:
        server.close()


def _force_proxy_environment(monkeypatch: pytest.MonkeyPatch, proxy_url: str) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, proxy_url)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")


def test_trusted_opener_ignores_proxy_for_credentialed_request(monkeypatch: pytest.MonkeyPatch) -> None:
    with _recording_server() as target, _recording_server() as proxy:
        _force_proxy_environment(monkeypatch, proxy.url)
        opener = build_trusted_opener(target.url)
        request = Request(
            f"{target.url}/signin",
            data=b'{"password":"test-admin-password"}',
            headers={"Authorization": "Bearer test-admin-jwt"},
            method="POST",
        )

        with opener(request, timeout=2) as response:
            assert response.status == 200

        assert proxy.records == []
        assert target.records == [
            ("/signin", "Bearer test-admin-jwt", b'{"password":"test-admin-password"}')
        ]


def test_pipeline_client_fails_closed_on_redirect_without_credential_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _recording_server() as redirector, _recording_server() as collector, _recording_server() as proxy:
        redirector.status = 302
        redirector.location = f"{collector.url}/capture"
        _force_proxy_environment(monkeypatch, proxy.url)
        monkeypatch.setattr(pipeline, "_TRUSTED_OPENWEBUI_INTERNAL_URL", redirector.url)
        client = OpenWebUIFileClient(
            base_url=redirector.url,
            email="admin@example.test",
            password="test-admin-password",
            timeout_seconds=2,
        )

        with pytest.raises(FileFetchError):
            client._sign_in()
        with pytest.raises(FileFetchError):
            client._get_file_with_token("f1d8f938-3c38-4f5f-a6d1-3c54e7cb5fc0", "test-admin-jwt")

        assert proxy.records == []
        assert collector.records == []
        assert redirector.records == [
            ("/api/v1/auths/signin", None, b'{"email": "admin@example.test", "password": "test-admin-password"}'),
            ("/api/v1/files/f1d8f938-3c38-4f5f-a6d1-3c54e7cb5fc0", "Bearer test-admin-jwt", b""),
        ]


def test_redirect_handler_returns_http_error_instead_of_following_location() -> None:
    with _recording_server() as redirector, _recording_server() as collector:
        redirector.status = 302
        redirector.location = f"{collector.url}/capture"
        opener = build_trusted_opener(redirector.url)

        with pytest.raises(HTTPError, match="redirects are not permitted"):
            opener(Request(f"{redirector.url}/redirect", method="GET"), timeout=2)

        assert collector.records == []


@pytest.mark.parametrize(
    "value",
    [
        "https://openwebui:8080",
        "http://openwebui:8080/redirect",
        "http://admin@openwebui:8080",
        "http://openwebui:8080?target=attacker.invalid",
        "http://attacker.invalid:8080",
    ],
)
def test_exact_base_url_rejects_untrusted_authorities(value: str) -> None:
    with pytest.raises(TrustedHttpError):
        require_exact_base_url(value, expected="http://openwebui:8080")
