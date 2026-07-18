"""Fetch-adjacent guard для закреплённого OpenWebUI image converter."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Coroutine, Mapping
from functools import wraps
from typing import Any, cast

REMOTE_IMAGE_URL_BLOCKED_DETAIL = "Remote image URLs disabled for Phase 0"

_PINNED_CONVERTER_SIGNATURE = "(form_data, user=None)"
_PATCH_MARKER = "__mtbank_remote_image_fetch_guard__"

Converter = Callable[..., Coroutine[Any, Any, Any]]


class RemoteImageURLBlockedError(RuntimeError):
    """Показывает, что remote image URL заблокирован до DNS и fetch."""


def install_remote_image_fetch_guard(middleware_module: Any) -> Converter:
    """Патчит module global converter, который вызывает pinned middleware flow."""

    original = getattr(middleware_module, "convert_url_images_to_base64", None)
    if is_remote_image_fetch_guard_installed(original):
        return cast(Converter, original)

    if not inspect.iscoroutinefunction(original) or str(inspect.signature(original)) != _PINNED_CONVERTER_SIGNATURE:
        raise RuntimeError("Unsupported OpenWebUI convert_url_images_to_base64 contract")

    original = cast(Converter, original)

    # OpenWebUI v0.10.2 revision ecd48e2f718220a6400ecf49eafd4867a38feb10
    # вызывает этот global после DB rehydrate/image-file conversion и до inlet.
    @wraps(original)
    async def guarded_convert_url_images_to_base64(form_data, user=None):
        if _contains_remote_image_fetch_candidate(form_data):
            raise RemoteImageURLBlockedError(REMOTE_IMAGE_URL_BLOCKED_DETAIL)
        return await original(form_data, user=user)

    setattr(guarded_convert_url_images_to_base64, _PATCH_MARKER, True)
    middleware_module.convert_url_images_to_base64 = guarded_convert_url_images_to_base64
    return guarded_convert_url_images_to_base64


def is_remote_image_fetch_guard_installed(converter: object) -> bool:
    """Проверяет marker без импорта OpenWebUI."""

    return getattr(converter, _PATCH_MARKER, False) is True


def _contains_remote_image_fetch_candidate(form_data: object) -> bool:
    if not isinstance(form_data, Mapping):
        return False

    messages = form_data.get("messages")
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, Mapping) and _is_remote_http_url(image_url.get("url")):
                return True
    return False


def _is_remote_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    scheme, separator, _ = value.strip().partition(":")
    return bool(separator) and scheme.casefold() in {"http", "https"}
