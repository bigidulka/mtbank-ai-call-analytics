"""Dual-media `/analyze` transport boundary."""

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import HttpUrl, ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartException

from mtbank_ai.api.dependencies import get_analyzer, get_request_id, get_settings, require_api_key
from mtbank_ai.api.schemas import UrlAnalyzeRequest
from mtbank_ai.application.ports import (
    AnalyzeCallPort,
    AnalyzeInput,
    FileAnalyzeInput,
    UrlAnalyzeInput,
)
from mtbank_ai.config import ApiSettings, Settings
from mtbank_ai.domain.analysis import AnalyzeResponse
from mtbank_ai.domain.errors import DomainError, ErrorCode, ErrorResponse

router = APIRouter()

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in (
        (400, "Источник отсутствует"),
        (401, "Требуется аутентификация"),
        (403, "Доступ запрещён"),
        (413, "Превышен допустимый размер"),
        (415, "Неподдерживаемый media type"),
        (422, "Некорректный URL, аудио или запрос"),
        (429, "Исчерпана квота"),
        (500, "Внутренняя ошибка"),
        (502, "Ошибка агента или обязательного провайдера"),
        (503, "Зависимость foundation недоступна"),
        (504, "Истёк deadline"),
    )
}

_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            },
            "application/json": {
                "schema": {"$ref": "#/components/schemas/UrlAnalyzeRequest"}
            },
        },
    }
}


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses=_ERROR_RESPONSES,
    openapi_extra=_REQUEST_BODY,
)
async def analyze(
    request: Request,
    analyzer: Annotated[AnalyzeCallPort, Depends(get_analyzer)],
    request_id: Annotated[UUID, Depends(get_request_id)],
    settings: Annotated[Settings, Depends(get_settings)],
    authenticated: Annotated[None, Depends(require_api_key)],
) -> AnalyzeResponse:
    del authenticated
    source = await _parse_source(request, settings.api)
    try:
        return await analyzer.analyze(source, request_id=request_id)
    except DomainError:
        raise
    except Exception:
        raise DomainError(ErrorCode.INTERNAL_ERROR) from None


async def _parse_source(request: Request, api_settings: ApiSettings) -> AnalyzeInput:
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().casefold()
    if media_type == "multipart/form-data":
        return await _parse_multipart(request, api_settings)
    if media_type == "application/json":
        return await _parse_json(request, api_settings)
    raise DomainError(ErrorCode.UNSUPPORTED_MEDIA)


async def _parse_multipart(request: Request, api_settings: ApiSettings) -> FileAnalyzeInput:
    form: FormData | None = None
    try:
        try:
            form = await request.form(
                max_files=2,
                max_fields=2,
                max_part_size=api_settings.max_upload_bytes + 1,
            )
        except (MultiPartException, StarletteHTTPException, ValueError) as error:
            if "maximum size" in str(error).casefold():
                raise DomainError(ErrorCode.PAYLOAD_TOO_LARGE) from error
            raise DomainError(ErrorCode.INVALID_REQUEST) from error

        assert form is not None
        items = form.multi_items()
        if not items:
            raise DomainError(ErrorCode.INVALID_INPUT)
        if len(items) != 1:
            raise DomainError(ErrorCode.INVALID_REQUEST)
        key, value = items[0]
        if key != "file" or not isinstance(value, UploadFile):
            raise DomainError(ErrorCode.INVALID_REQUEST)

        declared_media_type = (value.content_type or "").partition(";")[0].strip().casefold()
        if declared_media_type not in api_settings.allowed_media_types:
            raise DomainError(ErrorCode.UNSUPPORTED_MEDIA)

        filename = (value.filename or "").strip()
        content = await value.read(api_settings.max_upload_bytes + 1)
        if len(content) > api_settings.max_upload_bytes:
            raise DomainError(ErrorCode.PAYLOAD_TOO_LARGE)
        if not content or not filename:
            raise DomainError(ErrorCode.INVALID_AUDIO)
        return FileAnalyzeInput(
            filename=filename,
            content_type=declared_media_type,
            content=content,
        )
    finally:
        if form is not None:
            await form.close()


async def _parse_json(request: Request, api_settings: ApiSettings) -> UrlAnalyzeInput:
    try:
        data = await request.json()
    except (JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise DomainError(ErrorCode.INVALID_REQUEST) from error

    if not isinstance(data, Mapping):
        raise DomainError(ErrorCode.INVALID_REQUEST)
    if not data:
        raise DomainError(ErrorCode.INVALID_INPUT)
    if set(data) != {"url"}:
        raise DomainError(ErrorCode.INVALID_REQUEST)
    return UrlAnalyzeInput(url=_validate_url(data["url"], api_settings))


def _validate_url(value: object, api_settings: ApiSettings) -> HttpUrl:
    try:
        url = UrlAnalyzeRequest.model_validate({"url": value}).url
    except ValidationError as error:
        raise DomainError(ErrorCode.INVALID_URL) from error
    if url.scheme not in api_settings.allowed_url_schemes:
        raise DomainError(ErrorCode.INVALID_URL)
    return url
