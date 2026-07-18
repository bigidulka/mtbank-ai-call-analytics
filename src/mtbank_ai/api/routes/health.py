"""Process liveness и injected dependency readiness."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from mtbank_ai.api.dependencies import get_readiness
from mtbank_ai.api.schemas import HealthResponse
from mtbank_ai.application.ports import ReadinessPort
from mtbank_ai.domain.errors import DomainError, ErrorCode, ErrorResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": ErrorResponse, "description": "PostgreSQL недоступен"}},
)
async def ready(
    readiness: Annotated[ReadinessPort, Depends(get_readiness)],
) -> HealthResponse:
    try:
        is_ready = await readiness.ping()
    except Exception as error:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE) from error
    if not is_ready:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    return HealthResponse(status="ready")
