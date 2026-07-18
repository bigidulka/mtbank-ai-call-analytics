"""Публичные transport schemas, не зависящие от будущего workflow."""

from __future__ import annotations

from typing import Self

from pydantic import HttpUrl, model_validator

from mtbank_ai.domain.base import FrozenModel, NonEmptyId


class UrlAnalyzeRequest(FrozenModel):
    url: HttpUrl

    @model_validator(mode="after")
    def validate_url(self) -> Self:
        if self.url.scheme not in {"http", "https"}:
            raise ValueError("URL должен использовать HTTP или HTTPS")
        if self.url.username is not None or self.url.password is not None or self.url.fragment is not None:
            raise ValueError("URL не должен содержать credentials или fragment")
        return self


class HealthResponse(FrozenModel):
    status: NonEmptyId
