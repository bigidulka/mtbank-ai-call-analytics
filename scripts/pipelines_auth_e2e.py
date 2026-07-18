"""Проверяет Bearer-границу Pipelines через прямой container IP."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

_ENDPOINTS = (
    ("POST", "/v1/chat/completions"),
    ("POST", "/mtbank-attachment-probe/filter/inlet"),
    ("GET", "/mtbank-attachment-probe/valves"),
)
_WRONG_AUTHORIZATION = "Bearer this-is-deliberately-not-the-pipelines-key"


def _expect_unauthorized(base_url: str, method: str, path: str, authorization: str | None) -> None:
    headers: dict[str, str] = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=b"{}" if method == "POST" else None,
        headers=headers,
        method=method,
    )
    try:
        urlopen(request, timeout=10)
    except HTTPError as error:
        if error.code == 401:
            return
        raise RuntimeError(f"{method} {path} вернул HTTP {error.code} вместо 401") from error
    raise RuntimeError(f"{method} {path} принял неавторизованный запрос")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()

    for method, path in _ENDPOINTS:
        _expect_unauthorized(args.base_url, method, path, authorization=None)
        _expect_unauthorized(args.base_url, method, path, authorization=_WRONG_AUTHORIZATION)

    print(
        json.dumps(
            {
                "endpoints": [path for _, path in _ENDPOINTS],
                "unauthenticated": 401,
                "wrong_bearer": 401,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
