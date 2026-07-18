from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast
from urllib.request import OpenerDirector, Request

import pytest

from mtbank_ai.trusted_http import TrustedHttpError

ROOT = Path(__file__).parents[2]


def test_model_form_exposes_only_public_read_access() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts" / "model_capability_bootstrap.py"))

    model_form = cast(Callable[[], dict[str, object]], namespace["_model_form"])
    form = model_form()

    assert form["base_model_id"] is None
    assert form["access_grants"] == [
        {
            "principal_type": "user",
            "principal_id": "*",
            "permission": "read",
        }
    ]
    assert "write" not in str(form["access_grants"])


def test_bootstrap_rejects_untrusted_callback_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENWEBUI_BOOTSTRAP_URL", "http://attacker.invalid:8080")

    with pytest.raises(TrustedHttpError):
        runpy.run_path(str(ROOT / "scripts" / "model_capability_bootstrap.py"))


def test_bootstrap_uses_trusted_opener_for_bearer_request(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = runpy.run_path(str(ROOT / "scripts" / "model_capability_bootstrap.py"))
    captured: list[Request] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def geturl(self) -> str:
            return "http://openwebui:8080/api/models"

        def read(self) -> bytes:
            return b"{}"

    def open_request(self: OpenerDirector, request: Request, *, timeout: float) -> Response:
        del self, timeout
        captured.append(request)
        return Response()

    monkeypatch.setattr(OpenerDirector, "open", open_request)
    request_json = cast(Callable[..., object], namespace["_request_json"])
    response = request_json("/api/models", method="GET", token="test-admin-jwt")

    assert response == {}
    assert len(captured) == 1
    assert captured[0].get_header("Authorization") == "Bearer test-admin-jwt"
