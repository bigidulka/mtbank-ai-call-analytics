"""Запускает закреплённое Pipelines ASGI-приложение за общей Bearer-границей."""

from __future__ import annotations

import os
from typing import cast

from main import app as upstream_app

from mtbank_ai.pipeline_auth import ASGIApp, PipelineBearerAuth
from mtbank_ai.runtime_secrets import require_runtime_secret

api_key = require_runtime_secret("PIPELINES_API_KEY", os.getenv("PIPELINES_API_KEY"))
app = PipelineBearerAuth(cast(ASGIApp, upstream_app), api_key)
