"""Синхронный OpenWebUI adapter для shared async AnalyzeCallUseCase."""

from __future__ import annotations

import asyncio
import html
import json
import threading
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID

from mtbank_ai.application.ports import FileAnalyzeInput
from mtbank_ai.domain.analysis import AnalyzeResponse


class OpenWebUIAnalyzePort(Protocol):
    async def analyze_openwebui(self, source: FileAnalyzeInput, *, request_id: UUID) -> AnalyzeResponse: ...


class PipelineAnalysisPort(Protocol):
    def analyze(self, source: FileAnalyzeInput, *, request_id: UUID) -> AnalyzeResponse: ...


class OpenWebUIAnalysisAdapter:
    """Не создаёт второй workflow: вызывает тот же injected use case."""

    def __init__(self, analyzer: OpenWebUIAnalyzePort) -> None:
        self._analyzer = analyzer

    def analyze(self, source: FileAnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        return run_async_safely(self._analyzer.analyze_openwebui(source, request_id=request_id))


_Value = TypeVar("_Value")


@dataclass(slots=True)
class _ThreadResult(Generic[_Value]):
    value: _Value | None = None
    error: BaseException | None = None


def run_async_safely(awaitable: Coroutine[Any, Any, _Value]) -> _Value:
    """Выполняет async use case из sync `pipe`, в том числе внутри active event loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: _ThreadResult[_Value] = _ThreadResult()

    def run() -> None:
        try:
            result.value = asyncio.run(awaitable)
        except BaseException as error:
            result.error = error

    worker = threading.Thread(target=run, name="mtbank-openwebui-analysis", daemon=False)
    worker.start()
    worker.join()
    if result.error is not None:
        raise result.error
    if result.value is None:
        raise RuntimeError("async OpenWebUI adapter завершился без результата")
    return result.value


def render_openwebui_analysis(response: AnalyzeResponse, *, display_name: str) -> str:
    """Возвращает escaped canonical public DTO без raw attachment bytes или prompt traces."""

    rendered = json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return f"## {html.escape(display_name, quote=True)}\n\n<pre>{html.escape(rendered, quote=True)}</pre>"
