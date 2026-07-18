"""Typed доступ к injected app dependencies."""

from __future__ import annotations

import hmac
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from mtbank_ai.application.ports import AnalyzeCallPort, ReadinessPort
from mtbank_ai.config import Settings
from mtbank_ai.domain.errors import DomainError, ErrorCode

bearer_auth = HTTPBearer(auto_error=False, scheme_name="BearerAuth")


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_analyzer(request: Request) -> AnalyzeCallPort:
    return cast(AnalyzeCallPort, request.app.state.analyzer)


async def require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_auth)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise DomainError(ErrorCode.UNAUTHENTICATED)
    try:
        supplied_key = credentials.credentials.encode("ascii")
    except UnicodeEncodeError:
        raise DomainError(ErrorCode.UNAUTHENTICATED) from None
    expected_key = settings.api.api_key.get_secret_value().encode("ascii")
    if not hmac.compare_digest(supplied_key, expected_key):
        raise DomainError(ErrorCode.UNAUTHENTICATED)


def get_readiness(request: Request) -> ReadinessPort:
    return cast(ReadinessPort, request.app.state.readiness)


def get_request_id(request: Request) -> UUID:
    return cast(UUID, request.state.request_id)
