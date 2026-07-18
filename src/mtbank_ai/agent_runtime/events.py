"""Sanitized lifecycle event recording и hash-chained trajectory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from mtbank_ai.domain.events import EventAttribute, LifecycleEventType, RedactedPayload, RunEvent

_RAW_PAYLOAD_MARKERS = (
    "api_key",
    "argument",
    "authorization",
    "body",
    "content",
    "prompt",
    "response",
    "secret",
    "transcript",
    "value",
)
_SAFE_SENSITIVE_SUFFIXES = ("_bytes", "_code", "_hash", "_id")


class EventRedactionError(ValueError):
    """Payload пытается сохранить контент вместо разрешённой metadata."""


class EventSink(Protocol):
    async def append(self, event: RunEvent) -> None: ...


class NullEventSink:
    async def append(self, event: RunEvent) -> None:
        del event


class InMemoryEventSink:
    """Тестовый sink; production adapter обязан быть append-only."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append(self, event: RunEvent) -> None:
        self.events.append(event)


class LifecycleRecorder:
    """Пишет только scalar metadata, IDs, hashes, status и resource counters."""

    def __init__(
        self,
        *,
        run_id: UUID,
        sink: EventSink | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._run_id = run_id
        self._sink = sink or NullEventSink()
        self._now = now
        self._sequence = 0
        self._previous_hash: str | None = None

    async def record(
        self,
        event_type: LifecycleEventType,
        *,
        component: str = "agent_runtime",
        payload: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> RunEvent:
        _validate_payload(payload or {})
        self._sequence += 1
        redacted_payload = RedactedPayload(
            fields=tuple(EventAttribute(key=key, value=value) for key, value in sorted((payload or {}).items()))
        )
        occurred_at = self._now()
        current_hash = _event_hash(
            run_id=self._run_id,
            sequence=self._sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            component=component,
            payload=redacted_payload,
            previous_hash=self._previous_hash,
        )
        event = RunEvent(
            run_id=self._run_id,
            sequence=self._sequence,
            event_type=event_type,
            occurred_at=occurred_at,
            component=component,
            payload=redacted_payload,
            previous_hash=self._previous_hash,
            current_hash=current_hash,
        )
        await self._sink.append(event)
        self._previous_hash = current_hash
        return event


def _validate_payload(payload: Mapping[str, str | int | float | bool | None]) -> None:
    for key in payload:
        normalized = key.casefold()
        if any(marker in normalized for marker in _RAW_PAYLOAD_MARKERS) and not normalized.endswith(
            _SAFE_SENSITIVE_SUFFIXES
        ):
            raise EventRedactionError("event payload допускает только redacted metadata")


def _event_hash(
    *,
    run_id: UUID,
    sequence: int,
    event_type: LifecycleEventType,
    occurred_at: datetime,
    component: str,
    payload: RedactedPayload,
    previous_hash: str | None,
) -> str:
    document = {
        "component": component,
        "event_type": event_type.value,
        "occurred_at": occurred_at.isoformat(),
        "payload": payload.model_dump(mode="json"),
        "previous_hash": previous_hash,
        "run_id": str(run_id),
        "sequence": sequence,
    }
    encoded = json.dumps(document, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
