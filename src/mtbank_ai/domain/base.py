"""Общие неизменяемые типы доменных и публичных контрактов."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, Strict, StringConstraints


class FrozenModel(BaseModel):
    """Публичная frozen-модель с targeted validation без blanket strict mode."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class StrictFrozenModel(FrozenModel):
    """Внутренняя frozen-модель без неявных преобразований."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("временная метка должна быть timezone-aware UTC")
    return value


UtcDateTime = Annotated[datetime, AfterValidator(_require_utc)]
NonEmptyId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MimeType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        pattern=r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,126}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$",
    ),
]
Confidence = Annotated[float, Strict(), Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Strict(), Field(ge=0.0)]
PositiveFloat = Annotated[float, Strict(), Field(gt=0.0)]
NonNegativeInt = Annotated[int, Strict(), Field(ge=0)]
PositiveInt = Annotated[int, Strict(), Field(gt=0)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]
JsonScalar: TypeAlias = str | int | float | bool | None
ReasoningEffort: TypeAlias = Literal["low", "medium", "high", "xhigh", "ultra"]
