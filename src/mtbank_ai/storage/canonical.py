"""Каноническое JSON-представление для digest persistence boundary."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


def canonical_json_bytes(value: object) -> bytes:
    """Сериализует JSON-compatible value детерминированно без нечисловых значений."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
