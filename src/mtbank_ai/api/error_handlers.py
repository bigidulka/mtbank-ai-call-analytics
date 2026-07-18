"""Единые безопасные HTTP error responses."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from mtbank_ai.domain.errors import DomainError, ErrorCode, build_error_response

_HTTP_ERROR_CODES = {
    400: ErrorCode.INVALID_INPUT,
    401: ErrorCode.UNAUTHENTICATED,
    403: ErrorCode.FORBIDDEN,
    409: ErrorCode.ROLE_RESOLUTION_REQUIRED,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    415: ErrorCode.UNSUPPORTED_MEDIA,
    422: ErrorCode.INVALID_REQUEST,
    429: ErrorCode.QUOTA_EXCEEDED,
    502: ErrorCode.PROVIDER_FAILURE,
    503: ErrorCode.SERVICE_UNAVAILABLE,
    504: ErrorCode.DEADLINE_EXCEEDED,
}


def _request_id(request: Request) -> UUID:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, UUID) else uuid4()


def _json_error(request: Request, error: DomainError) -> JSONResponse:
    status_code, body = build_error_response(error, _request_id(request))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={"X-Request-ID": str(body.error.request_id)},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def domain_error_handler(request: Request, error: DomainError) -> JSONResponse:
        return _json_error(request, error)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
        del error
        return _json_error(request, DomainError(ErrorCode.INVALID_REQUEST))

    @app.exception_handler(ResponseValidationError)
    async def response_validation_error_handler(request: Request, error: ResponseValidationError) -> JSONResponse:
        del error
        return _json_error(request, DomainError(ErrorCode.INTERNAL_ERROR))

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException) -> Response:
        if error.status_code in {404, 405}:
            return await http_exception_handler(request, error)
        code = _HTTP_ERROR_CODES.get(error.status_code, ErrorCode.INVALID_REQUEST)
        return _json_error(request, DomainError(code))

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, error: Exception) -> JSONResponse:
        del error
        return _json_error(request, DomainError(ErrorCode.INTERNAL_ERROR))
