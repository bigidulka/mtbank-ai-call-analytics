"""Fail-closed urllib boundary for credential-bearing internal callbacks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import SplitResult, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class TrustedHttpError(ValueError):
    """Trusted internal HTTP origin invariant was violated."""


class FailClosedRedirectHandler(HTTPRedirectHandler):
    """Raises on every redirect before urllib can replay a credentialed request."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del msg, newurl
        raise HTTPError(req.full_url, code, "redirects are not permitted", headers, fp)


def require_exact_base_url(value: object, *, expected: str) -> str:
    """Accepts exactly one configured base URL with an explicit authority."""

    _parse_base_url(expected)
    if not isinstance(value, str) or value != expected:
        raise TrustedHttpError(f"trusted callback должен быть {expected}")
    _parse_base_url(value)
    return value


def build_trusted_opener(expected_base_url: str) -> Callable[..., Any]:
    """Builds an opener that ignores proxies, rejects redirects and pins its origin."""

    require_exact_base_url(expected_base_url, expected=expected_base_url)
    opener = build_opener(ProxyHandler({}), FailClosedRedirectHandler())

    def open_request(request: Request, *, timeout: float) -> Any:
        _require_origin(request.full_url, expected_base_url)
        response = opener.open(request, timeout=timeout)
        try:
            _require_origin(response.geturl(), expected_base_url)
        except BaseException:
            response.close()
            raise
        return response

    return open_request


def _parse_base_url(value: str) -> SplitResult:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise TrustedHttpError("trusted callback имеет некорректную authority") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or port is None
        or parsed.path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise TrustedHttpError("trusted callback должен быть base URL без path или credentials")
    return parsed


def _require_origin(value: object, expected_base_url: str) -> None:
    if not isinstance(value, str):
        raise TrustedHttpError("trusted HTTP URL отсутствует")
    expected = _parse_base_url(expected_base_url)
    try:
        actual = urlsplit(value)
        actual_port = actual.port
    except ValueError as error:
        raise TrustedHttpError("trusted HTTP URL имеет некорректную authority") from error
    if (
        actual.scheme != expected.scheme
        or actual.hostname != expected.hostname
        or actual_port != expected.port
        or actual.username is not None
        or actual.password is not None
    ):
        raise TrustedHttpError("trusted HTTP request покинул разрешённую authority")
