"""Protected bounded text-assistant JSON and SSE APIs."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Receive, Scope, Send

from mtbank_ai.agent_runtime import AgentFailureCode, AgentRuntimeError
from mtbank_ai.api.dependencies import require_api_key
from mtbank_ai.assistant import AssistantRequest, AssistantResponse, AssistantStreamEvent, DemoAssistant
from mtbank_ai.domain.errors import ERROR_SPECS, DomainError, ErrorCode

router = APIRouter()


class DisconnectAwareStreamingResponse(StreamingResponse):
    """Cancel stream iterator on disconnect under ASGI 2.4+ too."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        del scope
        stream_task = asyncio.create_task(self.stream_response(send))
        disconnect_task = asyncio.create_task(self.listen_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait(
                {stream_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done and not stream_task.done():
                stream_task.cancel()
            if stream_task in done and not disconnect_task.done():
                disconnect_task.cancel()
            results = await asyncio.gather(stream_task, disconnect_task, return_exceptions=True)
            stream_result = results[0]
            if isinstance(stream_result, BaseException) and not isinstance(
                stream_result, (asyncio.CancelledError, ClientDisconnect)
            ):
                raise stream_result
        finally:
            for task in (stream_task, disconnect_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stream_task, disconnect_task, return_exceptions=True)
        if self.background is not None:
            await self.background()


@router.post("/assistant", response_model=AssistantResponse)
async def assistant(
    request: AssistantRequest,
    app_request: Request,
    authenticated: Annotated[None, Depends(require_api_key)],
) -> AssistantResponse:
    del authenticated
    demo_assistant = _assistant(app_request)
    try:
        return await demo_assistant.answer(request)
    except AgentRuntimeError as error:
        raise DomainError(_map_agent_failure(error)) from None
    except Exception:
        raise DomainError(ErrorCode.AGENT_FAILURE) from None


@router.post("/assistant/stream")
async def assistant_stream(
    request: AssistantRequest,
    app_request: Request,
    authenticated: Annotated[None, Depends(require_api_key)],
) -> StreamingResponse:
    """SSE adapter with local sequence IDs and no provider/tool protocol leakage."""

    del authenticated
    demo_assistant = _assistant(app_request)

    async def events() -> AsyncIterator[str]:
        stream = demo_assistant.stream(request)
        last_sequence = 0
        try:
            async for event in stream:
                if await app_request.is_disconnected():
                    return
                last_sequence = event.sequence
                yield _sse(event.type, event.sequence, _public_event_payload(event))
        except asyncio.CancelledError:
            raise
        except AgentRuntimeError as error:
            sequence = last_sequence + 1
            yield _sse("error", sequence, _error_payload(_map_agent_failure(error), sequence))
        except Exception:
            sequence = last_sequence + 1
            yield _sse("error", sequence, _error_payload(ErrorCode.AGENT_FAILURE, sequence))
        finally:
            await cast(Any, stream).aclose()

    return DisconnectAwareStreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _assistant(app_request: Request) -> DemoAssistant:
    demo_assistant = cast(DemoAssistant | None, getattr(app_request.app.state, "demo_assistant", None))
    if demo_assistant is None:
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)
    return demo_assistant


def _sse(event: str, sequence: int, payload: dict[str, object]) -> str:
    return f"id: {sequence}\nevent: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _public_event_payload(event: AssistantStreamEvent) -> dict[str, object]:
    payload: dict[str, object] = {"v": 1, "sequence": event.sequence}
    if event.type == "progress":
        payload["phase"] = event.phase
        if event.tool_name is not None:
            payload["tool_name"] = event.tool_name
    elif event.type == "delta":
        payload["text"] = event.text
    return payload


def _error_payload(code: ErrorCode, sequence: int) -> dict[str, object]:
    spec = ERROR_SPECS[code]
    return {
        "v": 1,
        "sequence": sequence,
        "code": code.value,
        "message": spec.message,
        "retryable": spec.retryable,
    }


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
