"""Protected aggregate-only trend API."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request

from mtbank_ai.agent_runtime import AgentFailureCode, AgentRuntimeError
from mtbank_ai.api.dependencies import require_api_key
from mtbank_ai.domain.errors import DomainError, ErrorCode
from mtbank_ai.trends import TrendAnalysis, TrendRejected, TrendRequest, TrendsAgent

router = APIRouter()


@router.post("/trends", response_model=TrendAnalysis)
async def trends(
    request: TrendRequest,
    app_request: Request,
    authenticated: Annotated[None, Depends(require_api_key)],
) -> TrendAnalysis:
    del authenticated
    agent = cast(TrendsAgent | None, getattr(app_request.app.state, "trends_agent", None))
    if agent is None:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    try:
        return await agent.analyze(request)
    except TrendRejected:
        raise DomainError(ErrorCode.INVALID_INPUT) from None
    except AgentRuntimeError as error:
        raise DomainError(_map_agent_failure(error)) from None
    except Exception:
        raise DomainError(ErrorCode.AGENT_FAILURE) from None


def _map_agent_failure(error: AgentRuntimeError) -> ErrorCode:
    if error.code is AgentFailureCode.PROVIDER_RATE_LIMITED:
        return ErrorCode.QUOTA_EXCEEDED
    if error.code in {AgentFailureCode.DEADLINE_EXCEEDED, AgentFailureCode.PROVIDER_TIMEOUT}:
        return ErrorCode.DEADLINE_EXCEEDED
    if error.code is AgentFailureCode.CIRCUIT_OPEN:
        return ErrorCode.SERVICE_UNAVAILABLE
    if error.code in {
        AgentFailureCode.PROVIDER_AUTHENTICATION,
        AgentFailureCode.PROVIDER_PERMISSION,
        AgentFailureCode.PROVIDER_INVALID_REQUEST,
        AgentFailureCode.PROVIDER_TRANSPORT,
        AgentFailureCode.PROVIDER_SERVER,
    }:
        return ErrorCode.PROVIDER_FAILURE
    return ErrorCode.AGENT_FAILURE
