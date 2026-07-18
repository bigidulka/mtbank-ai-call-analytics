"""Проверка минимальных подписанных ссылок на вложения OpenWebUI."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, TypeGuard
from uuid import UUID

MAIN_PIPELINE_ID = "mtbank-attachment-probe"
REFERENCE_VERSION = 1
MAX_FUTURE_SKEW_SECONDS = 5
_MIN_SIGNING_KEY_BYTES = 32
_REFERENCE_FIELDS = frozenset({"v", "aud", "sub", "file_id", "iat", "exp", "signature"})


class _ReferencePayload(TypedDict):
    v: int
    aud: str
    sub: str
    file_id: str
    iat: int
    exp: int


class AttachmentBridgeError(ValueError):
    """Базовая ошибка данных, пересекающих attachment boundary."""


class AttachmentMetadataError(AttachmentBridgeError):
    """Некорректная client-side metadata файла OpenWebUI."""


class AttachmentReferenceError(AttachmentBridgeError):
    """Некорректная, просроченная или неавторизованная signed reference."""


@dataclass(frozen=True)
class AttachmentDescriptor:
    """Единственный допустимый клиентский признак — canonical UUID файла."""

    file_id: str


@dataclass(frozen=True)
class VerifiedAttachmentReference:
    """Проверенная transport reference без клиентских file hints."""

    file_id: str
    subject: str
    audience: str
    issued_at: int
    expires_at: int


def require_signing_key(value: str | bytes | None) -> bytes:
    """Возвращает достаточно длинный HMAC key, не раскрывая его значение."""

    if isinstance(value, str):
        key = value.encode("utf-8")
    elif isinstance(value, bytes):
        key = value
    else:
        raise AttachmentBridgeError("ключ подписи вложений не настроен")

    if len(key) < _MIN_SIGNING_KEY_BYTES:
        raise AttachmentBridgeError("ключ подписи вложений слишком короткий")
    return key


def extract_user_id(user: Mapping[str, object] | None) -> str:
    """Извлекает authenticated OpenWebUI user id из inlet envelope."""

    if not isinstance(user, Mapping):
        raise AttachmentMetadataError("контекст пользователя OpenWebUI отсутствует")
    return _normalise_text(user.get("id"), "идентификатор пользователя", max_length=256)


def normalise_file_descriptors(files: object) -> list[AttachmentDescriptor]:
    """Принимает ровно один direct browser item с ``type=file`` и UUID id."""

    if not _is_sequence(files):
        raise AttachmentMetadataError("metadata.files должен быть списком")
    if len(files) != 1:
        raise AttachmentMetadataError("Phase 0 поддерживает ровно одно прямое вложение")

    item = _as_string_mapping(files[0])
    if item is None:
        raise AttachmentMetadataError("элемент файла должен быть объектом")
    if item.get("type") != "file":
        raise AttachmentMetadataError("Phase 0 поддерживает только direct type=file")

    return [AttachmentDescriptor(file_id=_normalise_file_id(item.get("id")))]


def create_signed_reference(
    files: object,
    *,
    subject: str,
    audience: str,
    signing_key: str | bytes,
    issued_at: int | None = None,
    ttl_seconds: int,
) -> dict[str, object]:
    """Создаёт canonical reference только из UUID, audience и subject."""

    descriptor = normalise_file_descriptors(files)[0]
    iat = _normalise_timestamp(int(time.time()) if issued_at is None else issued_at)
    ttl = _normalise_ttl(ttl_seconds)
    payload: _ReferencePayload = {
        "v": REFERENCE_VERSION,
        "aud": _normalise_audience(audience),
        "sub": _normalise_text(subject, "subject", max_length=256),
        "file_id": descriptor.file_id,
        "iat": iat,
        "exp": iat + ttl,
    }
    signature = hmac.new(require_signing_key(signing_key), _canonical_json(payload), hashlib.sha256).hexdigest()
    return {**payload, "signature": signature}


def verify_signed_reference(
    reference: object,
    *,
    expected_subject: str,
    expected_audience: str,
    signing_key: str | bytes,
    now: int | None = None,
    max_ttl_seconds: int,
    max_future_skew_seconds: int = MAX_FUTURE_SKEW_SECONDS,
) -> VerifiedAttachmentReference:
    """Проверяет точный canonical payload, HMAC, audience, subject и время."""

    mapping = _as_string_mapping(reference)
    if mapping is None:
        raise AttachmentReferenceError("attachment reference должен быть объектом")
    if set(mapping) != _REFERENCE_FIELDS:
        raise AttachmentReferenceError("attachment reference содержит неподдерживаемые поля")

    payload = _normalise_reference_payload(mapping)
    supplied_signature = mapping.get("signature")
    if not isinstance(supplied_signature, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied_signature):
        raise AttachmentReferenceError("подпись attachment reference некорректна")

    expected_signature = hmac.new(
        require_signing_key(signing_key),
        _canonical_json(payload),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise AttachmentReferenceError("подпись attachment reference некорректна")

    subject = _normalise_text(expected_subject, "ожидаемый subject", max_length=256)
    audience = _normalise_audience(expected_audience)
    if payload["sub"] != subject:
        raise AttachmentReferenceError("attachment reference принадлежит другому пользователю")
    if payload["aud"] != audience:
        raise AttachmentReferenceError("audience attachment reference не совпадает")

    current_time = _normalise_timestamp(int(time.time()) if now is None else now)
    max_ttl = _normalise_ttl(max_ttl_seconds)
    if not isinstance(max_future_skew_seconds, int) or isinstance(max_future_skew_seconds, bool):
        raise AttachmentReferenceError("future skew attachment reference некорректен")
    if max_future_skew_seconds < 0:
        raise AttachmentReferenceError("future skew attachment reference некорректен")
    if payload["iat"] > current_time + max_future_skew_seconds:
        raise AttachmentReferenceError("attachment reference выпущен из будущего")
    if payload["exp"] <= payload["iat"] or payload["exp"] - payload["iat"] > max_ttl:
        raise AttachmentReferenceError("TTL attachment reference некорректен")
    if current_time >= payload["exp"]:
        raise AttachmentReferenceError("attachment reference просрочен")

    return VerifiedAttachmentReference(
        file_id=payload["file_id"],
        subject=payload["sub"],
        audience=payload["aud"],
        issued_at=payload["iat"],
        expires_at=payload["exp"],
    )


def _normalise_reference_payload(reference: Mapping[str, object]) -> _ReferencePayload:
    if reference.get("v") != REFERENCE_VERSION:
        raise AttachmentReferenceError("версия attachment reference не поддерживается")

    try:
        return {
            "v": REFERENCE_VERSION,
            "aud": _normalise_audience(reference.get("aud")),
            "sub": _normalise_text(reference.get("sub"), "subject", max_length=256),
            "file_id": _normalise_file_id(reference.get("file_id")),
            "iat": _normalise_timestamp(reference.get("iat")),
            "exp": _normalise_timestamp(reference.get("exp")),
        }
    except AttachmentBridgeError as error:
        raise AttachmentReferenceError("payload attachment reference некорректен") from error


def _as_string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None

    result: dict[str, object] = {}
    for key, item_value in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item_value
    return result


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalise_file_id(value: object) -> str:
    if not isinstance(value, str):
        raise AttachmentMetadataError("идентификатор файла должен быть UUID")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as error:
        raise AttachmentMetadataError("идентификатор файла должен быть canonical UUID") from error
    if str(parsed) != value:
        raise AttachmentMetadataError("идентификатор файла должен быть canonical UUID")
    return value


def _normalise_audience(value: object) -> str:
    return _normalise_text(value, "audience", max_length=128)


def _normalise_text(value: object, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise AttachmentMetadataError(f"{field} должен быть строкой")
    normalised = value.strip()
    if not normalised or len(normalised) > max_length:
        raise AttachmentMetadataError(f"длина {field} некорректна")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalised):
        raise AttachmentMetadataError(f"{field} содержит управляющие символы")
    return normalised


def _normalise_timestamp(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AttachmentReferenceError("временная метка attachment reference некорректна")
    return value


def _normalise_ttl(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 3600:
        raise AttachmentReferenceError("TTL attachment reference некорректен")
    return value


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
