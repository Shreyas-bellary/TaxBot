from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

import asyncpg
from pgvector.asyncpg import register_vector

from core.config import Settings, get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)


async def _init_connection(connection: asyncpg.Connection) -> None:
    """Per-connection setup hook: registers pgvector codecs."""

    try:
        await register_vector(connection)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("pgvector_codec_registration_failed", error=str(exc))


class Database:
    """Async Postgres facade backed by an ``asyncpg`` pool."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pool: asyncpg.Pool | None = None

    async def connect(self, *, min_size: int = 1, max_size: int = 10) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=str(self._settings.postgres_dsn),
            min_size=min_size,
            max_size=max_size,
            init=_init_connection,
            command_timeout=self._settings.irs_request_timeout_seconds,
        )
        logger.info("postgres_pool_ready", min_size=min_size, max_size=max_size)

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None
        logger.info("postgres_pool_closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database.connect() must be called before use")
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.acquire() as connection, connection.transaction():
            yield connection

    async def execute(self, query: str, *args: Any) -> str:
        async with self.acquire() as connection:
            return cast(str, await connection.execute(query, *args))

    async def fetch(self, query: str, *args: Any) -> Sequence[asyncpg.Record]:
        async with self.acquire() as connection:
            return cast(Sequence[asyncpg.Record], await connection.fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        async with self.acquire() as connection:
            return await connection.fetchrow(query, *args)

    async def __aenter__(self) -> Database:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
