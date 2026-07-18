from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from mtbank_ai.openwebui_image_guard import (
    REMOTE_IMAGE_URL_BLOCKED_DETAIL,
    RemoteImageURLBlockedError,
    install_remote_image_fetch_guard,
    is_remote_image_fetch_guard_installed,
)


def _fake_middleware() -> tuple[SimpleNamespace, list[tuple[object, object]]]:
    calls: list[tuple[object, object]] = []

    async def convert_url_images_to_base64(form_data, user=None):
        calls.append((form_data, user))
        return form_data

    return SimpleNamespace(convert_url_images_to_base64=convert_url_images_to_base64), calls


@pytest.mark.parametrize(
    "messages",
    [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/unbounded-image"},
                    }
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": " HTTP://example.invalid/unbounded-image "},
                    }
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Persisted image file."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://openwebui:8080/api/v1/files/file-id/content"},
                    },
                ],
            }
        ],
    ],
    ids=["https", "uppercase-http", "db-image-file-converted-to-image-url"],
)
def test_rejects_effective_remote_image_messages_before_original(messages: list[dict[str, Any]]) -> None:
    middleware, calls = _fake_middleware()
    converter = install_remote_image_fetch_guard(middleware)

    with pytest.raises(RemoteImageURLBlockedError) as error:
        asyncio.run(converter({"messages": messages}, user=object()))

    assert str(error.value) == REMOTE_IMAGE_URL_BLOCKED_DETAIL
    assert calls == []


@pytest.mark.parametrize(
    "form_data",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        }
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Обычный текст с https://example.invalid/image.png не является image_url.",
                }
            ]
        },
        {
            "messages": [{"role": "user", "content": "Проверь приложенный WAV."}],
            "files": [
                {
                    "type": "file",
                    "url": "https://example.invalid/audio.wav",
                    "content_type": "audio/wav",
                }
            ],
        },
    ],
    ids=["data-url", "ordinary-text", "top-level-audio-file"],
)
def test_preserves_original_converter_for_allowed_inputs(form_data: dict[str, Any]) -> None:
    middleware, calls = _fake_middleware()
    converter = install_remote_image_fetch_guard(middleware)
    user = object()

    result = asyncio.run(converter(form_data, user=user))

    assert result is form_data
    assert calls == [(form_data, user)]


def test_install_is_idempotent_and_keeps_exact_pinned_signature() -> None:
    middleware, calls = _fake_middleware()

    first = install_remote_image_fetch_guard(middleware)
    second = install_remote_image_fetch_guard(middleware)
    form_data = {"messages": [{"role": "user", "content": "Без изображения."}]}
    result = asyncio.run(second(form_data))

    assert first is second is middleware.convert_url_images_to_base64
    assert is_remote_image_fetch_guard_installed(second)
    assert str(inspect.signature(second, follow_wrapped=False)) == "(form_data, user=None)"
    assert result is form_data
    assert calls == [(form_data, None)]


def test_install_fails_closed_on_pinned_signature_drift() -> None:
    async def convert_url_images_to_base64(form_data):
        return form_data

    middleware = SimpleNamespace(convert_url_images_to_base64=convert_url_images_to_base64)

    with pytest.raises(RuntimeError, match="Unsupported OpenWebUI"):
        install_remote_image_fetch_guard(middleware)
