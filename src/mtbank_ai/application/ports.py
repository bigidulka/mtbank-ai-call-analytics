"""Минимальные application ports для transport foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias
from uuid import UUID

from pydantic import HttpUrl

from mtbank_ai.domain.analysis import AnalyzeResponse
from mtbank_ai.domain.errors import DomainError, ErrorCode


@dataclass(frozen=True, slots=True)
class FileAnalyzeInput:
    filename: str
    content_type: str
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class UrlAnalyzeInput:
    url: HttpUrl


AnalyzeInput: TypeAlias = FileAnalyzeInput | UrlAnalyzeInput


class AnalyzeCallPort(Protocol):
    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse: ...


class ReadinessPort(Protocol):
    async def ping(self) -> bool: ...


class UnavailableAnalyzeCall:
    async def analyze(self, source: AnalyzeInput, *, request_id: UUID) -> AnalyzeResponse:
        del source, request_id
        raise DomainError(ErrorCode.SERVICE_UNAVAILABLE)


class UnavailableReadiness:
    async def ping(self) -> bool:
        return False
