"""Fail-closed client port для configured speech service transport."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import HttpUrl, SecretStr, field_validator, model_validator

from mtbank_ai.config import (
    _SPEECH_MAX_SUCCESS_RESPONSE_BYTES,
    SpeechTransportMode,
    _require_speech_api_key,
    _validate_speech_base_url,
    _validate_speech_response_limits,
    _validate_speech_transcription_path,
)
from mtbank_ai.domain.base import PositiveFloat, PositiveInt, StrictFrozenModel
from mtbank_ai.domain.errors import DomainError, ErrorCode, ErrorResponse
from mtbank_ai.speech.contracts import SpeechFile, SpeechTranscriptionResponse

_SAFE_SPEECH_UPSTREAM_ERRORS = frozenset(
    {
        (409, ErrorCode.ROLE_RESOLUTION_REQUIRED),
        (413, ErrorCode.PAYLOAD_TOO_LARGE),
        (415, ErrorCode.UNSUPPORTED_MEDIA),
        (422, ErrorCode.INVALID_REQUEST),
        (422, ErrorCode.INVALID_AUDIO),
        (422, ErrorCode.NO_SPEECH),
        (429, ErrorCode.QUOTA_EXCEEDED),
        (502, ErrorCode.PROVIDER_FAILURE),
        (503, ErrorCode.SERVICE_UNAVAILABLE),
        (504, ErrorCode.DEADLINE_EXCEEDED),
    }
)


class SpeechTranscriptionPort(Protocol):
    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse: ...


class SpeechServiceClientSettings(StrictFrozenModel):
    mode: SpeechTransportMode = "internal_http"
    base_url: HttpUrl
    api_key: SecretStr | None = None
    transcription_path: str = "/v1/transcribe"
    timeout_seconds: PositiveFloat = 180.0
    max_success_response_bytes: PositiveInt = _SPEECH_MAX_SUCCESS_RESPONSE_BYTES
    max_error_response_bytes: PositiveInt = 16 * 1024

    @field_validator("transcription_path")
    @classmethod
    def validate_transcription_path(cls, value: str) -> str:
        return _validate_speech_transcription_path(value)

    @model_validator(mode="after")
    def validate_transport(self) -> SpeechServiceClientSettings:
        _validate_speech_base_url(self.base_url, self.mode)
        _require_speech_api_key(self.mode, self.api_key)
        _validate_speech_response_limits(self.max_success_response_bytes, self.max_error_response_bytes)
        return self


def _upstream_error_fallback(status_code: int) -> ErrorCode:
    if status_code in {401, 403} or status_code >= 500:
        return ErrorCode.PROVIDER_FAILURE
    return ErrorCode.SERVICE_UNAVAILABLE


def _sanitized_upstream_error(status_code: int, content: bytes) -> DomainError:
    fallback = _upstream_error_fallback(status_code)
    try:
        error = ErrorResponse.model_validate(json.loads(content)).error
    except (ValueError, json.JSONDecodeError):
        return DomainError(fallback)
    if (status_code, error.code) in _SAFE_SPEECH_UPSTREAM_ERRORS:
        return DomainError(error.code)
    return DomainError(fallback)


def _has_identity_content_encoding(response: httpx.Response) -> bool:
    content_encoding = response.headers.get("content-encoding")
    return content_encoding is None or content_encoding.strip().casefold() == "identity"


def _content_length_is_within_limit(response: httpx.Response, maximum_bytes: int) -> bool:
    content_length = response.headers.get("content-length")
    if content_length is None:
        return True
    normalized = content_length.strip()
    return normalized.isascii() and normalized.isdecimal() and int(normalized) <= maximum_bytes


async def _read_bounded_raw_content(response: httpx.Response, maximum_bytes: int) -> bytes | None:
    content = bytearray()
    async for chunk in response.aiter_raw():
        if len(chunk) > maximum_bytes - len(content):
            return None
        content.extend(chunk)
    return bytes(content)


def _response_limit_error(status_code: int) -> DomainError:
    if 200 <= status_code < 300:
        return DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    return DomainError(_upstream_error_fallback(status_code))


class HttpSpeechServiceClient:
    """Тонкий HTTP adapter без fallback между internal и remote providers."""

    def __init__(
        self,
        settings: SpeechServiceClientSettings,
        *,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory

    async def transcribe(self, source: SpeechFile) -> SpeechTranscriptionResponse:
        payload = source.metadata.model_dump(mode="json")
        files = {"file": (source.filename, source.content, source.content_type)}
        headers = {"Accept-Encoding": "identity"}
        if self._settings.mode == "remote_https":
            api_key = self._settings.api_key
            if api_key is None:
                raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
            headers["Authorization"] = f"Bearer {api_key.get_secret_value()}"
        try:
            async with self._client_factory(
                timeout=self._settings.timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{str(self._settings.base_url).rstrip('/')}{self._settings.transcription_path}",
                    data={"metadata": json.dumps(payload, separators=(",", ":"), ensure_ascii=False)},
                    files=files,
                    headers=headers,
                ) as response:
                    if not _has_identity_content_encoding(response):
                        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
                    if 300 <= response.status_code < 400:
                        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
                    success = 200 <= response.status_code < 300
                    maximum_bytes = (
                        self._settings.max_success_response_bytes
                        if success
                        else self._settings.max_error_response_bytes
                    )
                    if not _content_length_is_within_limit(response, maximum_bytes):
                        raise _response_limit_error(response.status_code)
                    content = await _read_bounded_raw_content(response, maximum_bytes)
                    if content is None:
                        raise _response_limit_error(response.status_code)
        except httpx.HTTPError as error:
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from error
        if not success:
            raise _sanitized_upstream_error(response.status_code, content)
        try:
            return SpeechTranscriptionResponse.model_validate_json(content)
        except ValueError as parse_error:
            raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from parse_error
