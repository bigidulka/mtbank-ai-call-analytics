"""Проверяет actual pinned middleware image sink внутри OpenWebUI container."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from collections.abc import Iterator, Mapping
from typing import Any

from mtbank_ai.openwebui_image_guard import (
    REMOTE_IMAGE_URL_BLOCKED_DETAIL,
    RemoteImageURLBlockedError,
    is_remote_image_fetch_guard_installed,
)


class _CountingFormData(Mapping[str, Any]):
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.messages_reads = 0

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "messages":
            self.messages_reads += 1
        return self._payload.get(key, default)


async def _check_contract() -> dict[str, object]:
    importlib.import_module("openwebui_wrapper")
    middleware = importlib.import_module("open_webui.utils.middleware")
    converter = middleware.convert_url_images_to_base64

    if not is_remote_image_fetch_guard_installed(converter):
        raise RuntimeError("Pinned OpenWebUI middleware image guard is not installed")
    if str(inspect.signature(converter, follow_wrapped=False)) != "(form_data, user=None)":
        raise RuntimeError("Pinned OpenWebUI middleware image converter signature drifted")

    remote_form = _CountingFormData(
        {
            "messages": [
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
            ]
        }
    )
    try:
        await converter(remote_form)
    except RemoteImageURLBlockedError as error:
        if str(error) != REMOTE_IMAGE_URL_BLOCKED_DETAIL:
            raise RuntimeError("Remote image rejection detail drifted") from error
    else:
        raise RuntimeError("Remote image URL reached the original converter")

    if remote_form.messages_reads != 1:
        raise RuntimeError("Remote image URL reached the original converter")

    allowed_forms = [
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
        {"messages": [{"role": "user", "content": "Обычный текст."}]},
        {
            "messages": [{"role": "user", "content": "Проверь приложенный WAV."}],
            "files": [{"type": "file", "content_type": "audio/wav"}],
        },
    ]
    for form_data in allowed_forms:
        if await converter(form_data) is not form_data:
            raise RuntimeError("Allowed form did not preserve the original converter result")

    return {
        "module_global_patched": True,
        "remote_image": "rejected before original converter",
        "allowed": ["data-url", "ordinary-text", "top-level-audio-file"],
    }


def main() -> None:
    print(json.dumps(asyncio.run(_check_contract()), separators=(",", ":")))


if __name__ == "__main__":
    main()
