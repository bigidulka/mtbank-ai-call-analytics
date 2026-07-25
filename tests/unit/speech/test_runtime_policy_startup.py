from __future__ import annotations

import asyncio

import httpx

from services.speech.app import create_app
from tests.unit.speech._helpers import make_registry


def test_app_readiness_fails_closed_when_role_agent_configuration_is_missing(tmp_path) -> None:
    _, settings = make_registry(tmp_path)
    invalid_settings = settings.model_copy(update={"role_agent": None})
    app = create_app(settings=invalid_settings)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://speech.test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_unavailable"

    asyncio.run(scenario())
