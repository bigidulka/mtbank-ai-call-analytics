"""Автоматически отключает file_context для закреплённой external Pipeline модели."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from mtbank_ai.trusted_http import TrustedHttpError, build_trusted_opener, require_exact_base_url

MAIN_PIPELINE_ID = "mtbank-attachment-probe"
_TRUSTED_OPENWEBUI_INTERNAL_URL = "http://openwebui:8080"
BASE_URL = require_exact_base_url(
    os.getenv("OPENWEBUI_BOOTSTRAP_URL", _TRUSTED_OPENWEBUI_INTERNAL_URL).rstrip("/"),
    expected=_TRUSTED_OPENWEBUI_INTERNAL_URL,
)
_TRUSTED_OPENER = build_trusted_opener(BASE_URL)
_TIMEOUT_SECONDS = 10
_MAX_DISCOVERY_ATTEMPTS = 30


class BootstrapError(RuntimeError):
    """Не удалось автоматически применить безопасную capability модели."""


class _HttpFailure(BootstrapError):
    """OpenWebUI API вернул неуспешный HTTP status."""

    def __init__(self, status: int) -> None:
        super().__init__(f"OpenWebUI вернул HTTP {status}")
        self.status = status


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise BootstrapError(f"{name} обязателен для model bootstrap")
    return value


def _request_json(
    path: str,
    *,
    method: str,
    token: str | None = None,
    payload: object | None = None,
) -> object:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with _TRUSTED_OPENER(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise _HttpFailure(error.code) from error
    except (URLError, OSError, TrustedHttpError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"не удалось вызвать OpenWebUI {path}") from error


def _sign_in() -> str:
    response = _request_json(
        "/api/v1/auths/signin",
        method="POST",
        payload={
            "email": _required_env("WEBUI_ADMIN_EMAIL"),
            "password": _required_env("WEBUI_ADMIN_PASSWORD"),
        },
    )
    if not isinstance(response, dict):
        raise BootstrapError("sign-in OpenWebUI вернул некорректный ответ")
    token = response.get("token")
    if not isinstance(token, str) or not token:
        raise BootstrapError("sign-in OpenWebUI не вернул JWT")
    return token


def _wait_for_pipeline_model(token: str) -> None:
    for attempt in range(_MAX_DISCOVERY_ATTEMPTS):
        try:
            response = _request_json("/api/models", method="GET", token=token)
        except BootstrapError:
            if attempt == _MAX_DISCOVERY_ATTEMPTS - 1:
                raise
            time.sleep(2)
            continue
        if isinstance(response, dict):
            models = response.get("data")
            if isinstance(models, list) and any(
                isinstance(model, dict) and model.get("id") == MAIN_PIPELINE_ID for model in models
            ):
                return
        if attempt == _MAX_DISCOVERY_ATTEMPTS - 1:
            raise BootstrapError("закреплённая Pipeline модель не обнаружена")
        time.sleep(2)


def _model_form() -> dict[str, object]:
    return {
        "id": MAIN_PIPELINE_ID,
        "base_model_id": None,
        "name": "MTBank Attachment Probe",
        "meta": {"capabilities": {"file_context": False}},
        "params": {},
        "access_grants": [
            {
                "principal_type": "user",
                "principal_id": "*",
                "permission": "read",
            }
        ],
        "is_active": True,
    }


def _upsert_model(token: str) -> dict[str, Any]:
    try:
        response = _request_json("/api/v1/models/create", method="POST", token=token, payload=_model_form())
    except _HttpFailure as error:
        if error.status != 401:
            raise
        response = _request_json("/api/v1/models/model/update", method="POST", token=token, payload=_model_form())

    if not isinstance(response, dict):
        raise BootstrapError("model API вернул некорректный ответ")
    return response


def _assert_file_context_disabled(model: dict[str, Any]) -> None:
    meta = model.get("meta")
    capabilities = meta.get("capabilities") if isinstance(meta, dict) else None
    if not isinstance(capabilities, dict) or capabilities.get("file_context") is not False:
        raise BootstrapError("model API не сохранил file_context=false")


def main() -> None:
    token = _sign_in()
    _wait_for_pipeline_model(token)
    model = _upsert_model(token)
    _assert_file_context_disabled(model)
    print(json.dumps({"model": MAIN_PIPELINE_ID, "file_context": False}))


if __name__ == "__main__":
    main()
