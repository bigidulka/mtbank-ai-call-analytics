"""Protected app-plane observation of the configured remote speech runtime."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends

from mtbank_ai.api.dependencies import get_settings, require_api_key
from mtbank_ai.config import Settings
from mtbank_ai.domain.errors import DomainError, ErrorCode

router = APIRouter(prefix="/v1", tags=["release"])
_MAX_RUNTIME_RESPONSE_BYTES = 32 * 1024
_RUNTIME_KEYS = frozenset({"device", "compute_type", "image_digest", "asr", "diarization"})
_COMPONENT_KEYS = frozenset({"package", "package_version", "model_id", "model_revision"})


@router.get("/benchmark-runtime-binding", dependencies=[Depends(require_api_key)])
async def benchmark_runtime_binding(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    """Returns a bounded observation fetched from the speech backend configured by this app."""

    speech = settings.speech
    if speech is None or speech.mode != "remote_https" or speech.api_key is None:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    runtime_url = f"{str(speech.base_url).rstrip('/')}/v1/runtime"
    headers = {"Accept-Encoding": "identity", "Authorization": f"Bearer {speech.api_key.get_secret_value()}"}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(speech.timeout_seconds), trust_env=False, follow_redirects=False
        ) as client:
            async with client.stream("GET", runtime_url, headers=headers) as response:
                if response.status_code != 200 or 300 <= response.status_code < 400:
                    raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
                content = await _read_bounded(response)
    except httpx.HTTPError as error:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from error
    runtime = _runtime_from_content(content)
    return {
        "schema_version": "1",
        "kind": "configured-remote-speech-runtime",
        "speech_backend_url_sha256": hashlib.sha256(str(speech.base_url).encode("utf-8")).hexdigest(),
        "runtime": runtime,
    }


async def _read_bounded(response: httpx.Response) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None and (
        not content_length.isdecimal() or int(content_length) > _MAX_RUNTIME_RESPONSE_BYTES
    ):
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    content = bytearray()
    async for chunk in response.aiter_raw():
        if len(chunk) > _MAX_RUNTIME_RESPONSE_BYTES - len(content):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
        content.extend(chunk)
    return bytes(content)


def _runtime_from_content(content: bytes) -> dict[str, object]:
    try:
        payload = httpx.Response(200, content=content).json()
    except ValueError as error:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from error
    if not isinstance(payload, Mapping) or set(payload) != {"runtime"} or not isinstance(payload["runtime"], Mapping):
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    runtime = payload["runtime"]
    if (
        set(runtime) != _RUNTIME_KEYS
        or not isinstance(runtime["device"], str)
        or not isinstance(runtime["compute_type"], str)
    ):
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    for component in ("asr", "diarization"):
        value = runtime[component]
        if (
            not isinstance(value, Mapping)
            or set(value) != _COMPONENT_KEYS
            or not all(isinstance(item, str) and item.strip() for item in value.values())
        ):
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    if not isinstance(runtime["image_digest"], str):
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    return dict(runtime)
