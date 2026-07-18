"""Async repository ports; concrete persistence появится вместе с workflow."""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from mtbank_ai.domain.analysis import SanitizedAnalysisRecord
from mtbank_ai.domain.errors import ErrorCode
from mtbank_ai.domain.events import RunEvent, RunStatus
from mtbank_ai.evidence.envelope import RunEnvelope

_Model = TypeVar("_Model", bound=BaseModel)


class RunRepository(Protocol):
    async def create(self, envelope: RunEnvelope) -> None: ...

    async def get(self, run_id: UUID) -> RunEnvelope | None: ...

    async def set_status(
        self,
        run_id: UUID,
        status: RunStatus,
        *,
        error_code: ErrorCode | None = None,
    ) -> None: ...


class EventRepository(Protocol):
    async def append(self, event: RunEvent) -> None: ...

    async def list(self, run_id: UUID) -> tuple[RunEvent, ...]: ...


class AnalysisRepository(Protocol):
    async def save_sanitized(self, record: SanitizedAnalysisRecord) -> None: ...

    async def get(self, run_id: UUID) -> SanitizedAnalysisRecord | None: ...


class AsyncUnitOfWork(Protocol):
    runs: RunRepository
    events: EventRepository
    analyses: AnalysisRepository

    async def __aenter__(self) -> AsyncUnitOfWork: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class SqlAlchemyRunRepository:
    """SQLAlchemy Core adapter, сохраняющий только RunEnvelope JSONB."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, envelope: RunEnvelope) -> None:
        from sqlalchemy import insert

        from mtbank_ai.domain.events import RunStatus
        from mtbank_ai.storage.models import runs

        await self._session.execute(
            insert(runs).values(
                run_id=envelope.run_id,
                request_id=envelope.request_id,
                correlation_id=envelope.correlation_id,
                source=envelope.source.value,
                status=RunStatus.QUEUED.value,
                envelope=envelope.model_dump(mode="json"),
                error_code=None,
            )
        )

    async def get(self, run_id: UUID) -> RunEnvelope | None:
        from sqlalchemy import select

        from mtbank_ai.storage.models import runs

        statement = select(runs.c.envelope).where(runs.c.run_id == run_id)
        payload = (await self._session.execute(statement)).scalar_one_or_none()
        return _model_from_json(RunEnvelope, payload)

    async def set_status(
        self,
        run_id: UUID,
        status: RunStatus,
        *,
        error_code: ErrorCode | None = None,
    ) -> None:
        from sqlalchemy import func, update

        from mtbank_ai.storage.models import runs

        await self._session.execute(
            update(runs)
            .where(runs.c.run_id == run_id)
            .values(status=status.value, error_code=error_code.value if error_code else None, updated_at=func.now())
        )


class SqlAlchemyEventRepository:
    """Append-only adapter; public API не предоставляет mutation/delete."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append(self, event: RunEvent) -> None:
        from sqlalchemy import insert

        from mtbank_ai.storage.models import run_events

        await self._session.execute(
            insert(run_events).values(
                run_id=event.run_id,
                sequence=event.sequence,
                event_type=event.event_type.value,
                occurred_at=event.occurred_at,
                component=event.component,
                redacted_payload=event.payload.model_dump(mode="json"),
                previous_hash=event.previous_hash,
                current_hash=event.current_hash,
            )
        )

    async def list(self, run_id: UUID) -> tuple[RunEvent, ...]:
        from sqlalchemy import select

        from mtbank_ai.storage.models import run_events

        rows = (
            await self._session.execute(
                select(run_events).where(run_events.c.run_id == run_id).order_by(run_events.c.sequence.asc())
            )
        ).mappings()
        events = tuple(
            _model_from_json(
                RunEvent,
                {
                    "run_id": row["run_id"],
                    "sequence": row["sequence"],
                    "event_type": row["event_type"],
                    "occurred_at": row["occurred_at"],
                    "component": row["component"],
                    "payload": row["redacted_payload"],
                    "previous_hash": row["previous_hash"],
                    "current_hash": row["current_hash"],
                },
            )
            for row in rows
        )
        if any(event is None for event in events):
            raise ValueError("persisted event отсутствует")
        return tuple(event for event in events if event is not None)


class SqlAlchemyAnalysisRepository:
    """Сохраняет только SanitizedAnalysisRecord и canonical digest."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_sanitized(self, record: SanitizedAnalysisRecord) -> None:
        from sqlalchemy import insert

        from mtbank_ai.storage.canonical import canonical_json_sha256
        from mtbank_ai.storage.models import analyses

        await self._session.execute(
            insert(analyses).values(
                run_id=record.run_id,
                sanitized_result=record.model_dump(mode="json"),
                sanitized_result_sha256=canonical_json_sha256(record),
            )
        )

    async def get(self, run_id: UUID) -> SanitizedAnalysisRecord | None:
        from sqlalchemy import select

        from mtbank_ai.storage.models import analyses

        payload = (
            await self._session.execute(select(analyses.c.sanitized_result).where(analyses.c.run_id == run_id))
        ).scalar_one_or_none()
        return _model_from_json(SanitizedAnalysisRecord, payload)


class SqlAlchemyUnitOfWork:
    """Один transaction boundary для envelope, events и sanitized analysis."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self.runs: RunRepository
        self.events: EventRepository
        self.analyses: AnalysisRepository

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        self._session = self._session_factory()
        self.runs = SqlAlchemyRunRepository(self._session)
        self.events = SqlAlchemyEventRepository(self._session)
        self.analyses = SqlAlchemyAnalysisRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        del exc_value, traceback
        if self._session is None:
            return None
        try:
            if exc_type is not None:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None
        return None

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work не открыт")
        await self._session.commit()

    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work не открыт")
        await self._session.rollback()


def create_sqlalchemy_uow_factory(engine: AsyncEngine) -> Callable[[], SqlAlchemyUnitOfWork]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return lambda: SqlAlchemyUnitOfWork(session_factory)


class SqlAlchemyTrendRepository:
    """Read-only parameterized query seam for sanitized aggregate trend evidence."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_sanitized(self, *, window_start, window_end) -> tuple[SanitizedAnalysisRecord, ...]:  # type: ignore[no-untyped-def]
        from sqlalchemy import select

        from mtbank_ai.storage.models import analyses, runs

        async with self._session_factory() as session:
            payloads = (
                await session.execute(
                    select(analyses.c.sanitized_result)
                    .join(runs, runs.c.run_id == analyses.c.run_id)
                    .where(runs.c.created_at >= window_start, runs.c.created_at < window_end)
                    .order_by(runs.c.created_at.asc())
                )
            ).scalars()
            records = tuple(_model_from_json(SanitizedAnalysisRecord, payload) for payload in payloads)
        if any(record is None for record in records):
            raise ValueError("sanitized trend record отсутствует")
        return tuple(record for record in records if record is not None)


def create_sqlalchemy_trend_repository(engine: AsyncEngine) -> SqlAlchemyTrendRepository:
    return SqlAlchemyTrendRepository(async_sessionmaker(engine, expire_on_commit=False))


def _model_from_json(model: type[_Model], payload: object) -> _Model | None:
    if payload is None:
        return None
    import json

    if not isinstance(payload, dict):
        raise ValueError("persisted JSONB должен быть object")
    return model.model_validate_json(json.dumps(payload, ensure_ascii=False, default=_json_default), strict=True)


def _json_default(value: object) -> str:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        result = isoformat()
        if isinstance(result, str):
            return result
    return str(value)
