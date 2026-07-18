"""Проверяет runtime logs без вывода потенциально чувствительного содержимого."""

from __future__ import annotations

import json
import re
import sys

_MAX_LOG_BYTES = 2 * 1024 * 1024
_SENSITIVE_MARKERS = (
    "authorization",
    "bearer ",
    "pipeline input",
    "request body",
    "form_data",
    "проверь приложенный wav",
    "attachment-positive.wav",
    "attachment-foreign.wav",
    "https://example.invalid/unbounded-image",
    "https:attacker.invalid",
)
_ACCESS_REQUEST_LINE = re.compile(r'"\S+ [^"\r\n]+ HTTP/\d(?:\.\d)?" \d{3}(?:\s|$)')
_NGINX_ERROR_REQUEST_LINE = re.compile(r'\brequest:\s*"\S+ [^"\r\n]+ HTTP/\d(?:\.\d)?"')


def _contains_sensitive_marker(logs: str) -> bool:
    normalised = logs.casefold()
    return any(marker in normalised for marker in _SENSITIVE_MARKERS)


def _contains_access_request_line(logs: str) -> bool:
    return _ACCESS_REQUEST_LINE.search(logs) is not None


def _contains_nginx_error_request_line(logs: str) -> bool:
    return _NGINX_ERROR_REQUEST_LINE.search(logs) is not None


def main() -> None:
    logs = sys.stdin.read(_MAX_LOG_BYTES + 1)
    if len(logs.encode("utf-8")) > _MAX_LOG_BYTES:
        raise SystemExit("runtime logs превышают безопасный лимит проверки")
    if _contains_sensitive_marker(logs):
        raise SystemExit("runtime logs содержат privacy-sensitive marker")
    if _contains_access_request_line(logs) or _contains_nginx_error_request_line(logs):
        raise SystemExit("runtime logs содержат request metadata")
    print(json.dumps({"runtime_logs": "privacy markers и request metadata отсутствуют"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
