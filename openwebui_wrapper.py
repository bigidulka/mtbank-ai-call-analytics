"""Runs the pinned OpenWebUI ASGI app behind the Phase 0 pre-body boundary."""

from __future__ import annotations

from open_webui.main import app as upstream_app
from open_webui.utils import middleware as openwebui_middleware

from mtbank_ai.openwebui_guard import OpenWebUIPreBodyGuard
from mtbank_ai.openwebui_image_guard import install_remote_image_fetch_guard

install_remote_image_fetch_guard(openwebui_middleware)
app = OpenWebUIPreBodyGuard(upstream_app)
