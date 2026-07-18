"""Создание async PostgreSQL engine без import-time подключения."""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mtbank_ai.config import DatabaseSettings, Settings


def build_postgres_url(database: DatabaseSettings) -> URL:
    return URL.create(
        drivername="postgresql+asyncpg",
        username=database.user,
        password=database.password.get_secret_value(),
        host=database.host,
        port=database.port,
        database=database.name,
    )


def create_postgres_engine(settings: Settings) -> AsyncEngine:
    """Создаёт pool; реальное соединение открывается только при первом запросе."""

    return create_async_engine(
        build_postgres_url(settings.database),
        connect_args={
            "timeout": settings.database.connect_timeout_seconds,
            "command_timeout": settings.database.command_timeout_seconds,
        },
        pool_pre_ping=True,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout_seconds,
    )


async def ping_postgres(engine: AsyncEngine) -> bool:
    async with engine.connect() as connection:
        value = await connection.scalar(text("SELECT 1"))
    return value == 1


class PostgresReadiness:
    def __init__(self, engine: AsyncEngine, readiness_timeout_seconds: float) -> None:
        self._engine = engine
        self._readiness_timeout_seconds = readiness_timeout_seconds

    async def ping(self) -> bool:
        async with asyncio.timeout(self._readiness_timeout_seconds):
            return await ping_postgres(self._engine)

    async def close(self) -> None:
        await self._engine.dispose()
