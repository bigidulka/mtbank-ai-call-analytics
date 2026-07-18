from __future__ import annotations

import asyncio

import httpx
import pytest

from mtbank_ai.policies import PolicyLoadError
from services.speech import runtime as speech_runtime
from services.speech.app import create_app
from tests.unit.speech._helpers import make_registry


def test_app_readiness_fails_closed_when_default_roles_policy_is_invalid(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, settings = make_registry(tmp_path)

    class InvalidPolicyRegistry:
        @property
        def roles(self):
            raise PolicyLoadError("invalid")

    monkeypatch.setattr(speech_runtime, "PolicyRegistry", InvalidPolicyRegistry)
    app = create_app(settings=settings)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://speech.test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"

    asyncio.run(scenario())
