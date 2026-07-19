"""Postgres-backed per-IP daily rate limiter.

Tracks ``(client_ip, UTC date) → answer count`` in the
``private.ip_daily_rate_limits`` Postgres table via three atomic database
functions:

* ``private.ip_rl_consume``  - reserve one slot (SELECT FOR UPDATE + increment)
* ``private.ip_rl_check``    - read-only snapshot
* ``private.ip_rl_refund``   - decrement on request failure

All three functions are called inside the existing asyncpg connection pool so
no additional connections or locks are needed in application code.  Because
the per-row lock is held inside the database the quota is consistent across
every Cloud Run instance sharing the same Supabase Postgres cluster.

Contrast with the previous ``asyncio.Lock``-protected in-process dict which
reset on restart and gave each worker its own independent quota.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from core.db import Database
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of a rate-limit check or consume call."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime
    retry_after_seconds: int


class IpDailyRateLimiter:
    """Allow at most ``limit`` successful answer slots per IP per UTC day.

    State is stored in ``private.ip_daily_rate_limits`` via atomic PostgreSQL
    functions so the quota is shared and consistent across all application
    instances.  Pass the connected :class:`~core.db.Database` pool at
    construction; it must already have had ``connect()`` called on it.
    """

    def __init__(self, *, limit: int = 3, database: Database) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._db = database

    @property
    def limit(self) -> int:
        return self._limit

    # ------------------------------------------------------------------
    # Public interface (mirrors the old in-memory API exactly)
    # ------------------------------------------------------------------

    async def check(self, client_ip: str) -> RateLimitDecision:
        """Return the caller's current quota state without consuming a slot."""

        row = await self._db.fetchrow(
            "SELECT allowed, answer_count, limit_val, reset_at "
            "FROM private.ip_rl_check($1, $2)",
            client_ip,
            self._limit,
        )
        return self._decision_from_row(row, client_ip)

    async def consume(self, client_ip: str) -> RateLimitDecision:
        """Reserve one answer slot.  Returns ``allowed=False`` when exhausted."""

        row = await self._db.fetchrow(
            "SELECT allowed, answer_count, limit_val, reset_at "
            "FROM private.ip_rl_consume($1, $2)",
            client_ip,
            self._limit,
        )
        decision = self._decision_from_row(row, client_ip)
        if decision.allowed:
            logger.info(
                "rate_limit_consumed",
                client_ip=client_ip,
                used=decision.limit - decision.remaining,
                remaining=decision.remaining,
                limit=decision.limit,
            )
        else:
            logger.info(
                "rate_limit_exceeded",
                client_ip=client_ip,
                limit=decision.limit,
                remaining=0,
            )
        return decision

    async def refund(self, client_ip: str) -> None:
        """Return one reserved slot (e.g. when the request failed before an answer)."""

        await self._db.execute(
            "SELECT private.ip_rl_refund($1)",
            client_ip,
        )
        logger.info("rate_limit_refunded", client_ip=client_ip)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_at(self, *, day: date | None = None) -> datetime:
        """UTC midnight that starts the day after ``day``."""
        target = day or datetime.now(UTC).date()
        next_day = target + timedelta(days=1)
        return datetime.combine(next_day, time.min, tzinfo=UTC)

    def _decision_from_row(
        self,
        row: object,
        client_ip: str,
    ) -> RateLimitDecision:
        """Convert a database function result row into a ``RateLimitDecision``."""

        if row is None:
            # Should never happen unless the migration has not been applied;
            # fail loudly so this is caught in staging before production.
            raise RuntimeError(
                "ip_rl_check / ip_rl_consume returned no row — "
                "ensure migration 002_ip_daily_rate_limits.sql has been applied."
            )

        # asyncpg Record supports attribute-style access.
        allowed: bool = row["allowed"]  # type: ignore[index]
        answer_count: int = row["answer_count"]  # type: ignore[index]
        limit_val: int = row["limit_val"]  # type: ignore[index]
        db_reset_at: datetime = row["reset_at"]  # type: ignore[index]

        # Ensure reset_at is always timezone-aware (asyncpg returns tz-aware
        # timestamptz values, but be defensive).
        if db_reset_at.tzinfo is None:
            db_reset_at = db_reset_at.replace(tzinfo=UTC)

        remaining = max(0, limit_val - answer_count)
        now = datetime.now(UTC)
        retry_after = max(0, int((db_reset_at - now).total_seconds())) if not allowed else 0

        return RateLimitDecision(
            allowed=allowed,
            limit=limit_val,
            remaining=remaining,
            reset_at=db_reset_at,
            retry_after_seconds=retry_after,
        )
