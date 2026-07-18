"""In-memory per-IP daily rate limiter.

Tracks only ``(client_ip, UTC date) → answer count``. State is maintained
in process memory and resets on application restart. This implementation is
intended for single-instance deployments where lightweight, best-effort
rate limiting is sufficient.

For multi-instance or horizontally scaled deployments, replace this with a
shared backend (e.g. Redis) to ensure consistent rate limits across all
instances.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of a rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime
    retry_after_seconds: int


class IpDailyRateLimiter:
    """Allow at most ``limit`` successful answer slots per IP per UTC day."""

    def __init__(self, *, limit: int = 3) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self._limit = limit
        self._lock = asyncio.Lock()
        # key: (ip, iso_date) → count of consumed answer slots today
        self._counts: dict[tuple[str, str], int] = {}

    @property
    def limit(self) -> int:
        return self._limit

    def _today_key(self, client_ip: str, *, now: datetime | None = None) -> tuple[str, str]:
        moment = now or datetime.now(UTC)
        return (client_ip, moment.date().isoformat())

    def _reset_at(self, *, day: date | None = None) -> datetime:
        target = day or datetime.now(UTC).date()
        next_day = target + timedelta(days=1)
        return datetime.combine(next_day, time.min, tzinfo=UTC)

    def _prune(self, *, keep_day: str) -> None:
        stale = [key for key in self._counts if key[1] != keep_day]
        for key in stale:
            del self._counts[key]

    async def check(self, client_ip: str) -> RateLimitDecision:
        """Return whether the IP still has free answer slots today (no consume)."""

        async with self._lock:
            return self._snapshot(client_ip)

    async def consume(self, client_ip: str) -> RateLimitDecision:
        """Reserve one answer slot. Returns ``allowed=False`` when exhausted."""

        async with self._lock:
            decision = self._snapshot(client_ip)
            if not decision.allowed:
                logger.info(
                    "rate_limit_exceeded",
                    client_ip=client_ip,
                    limit=self._limit,
                    remaining=0,
                )
                return decision

            key = self._today_key(client_ip)
            self._counts[key] = self._counts.get(key, 0) + 1
            used = self._counts[key]
            remaining = max(0, self._limit - used)
            reset_at = self._reset_at()
            logger.info(
                "rate_limit_consumed",
                client_ip=client_ip,
                used=used,
                remaining=remaining,
                limit=self._limit,
            )
            return RateLimitDecision(
                allowed=True,
                limit=self._limit,
                remaining=remaining,
                reset_at=reset_at,
                retry_after_seconds=0,
            )

    async def refund(self, client_ip: str) -> None:
        """Return one reserved slot (e.g. when the request failed before an answer)."""

        async with self._lock:
            key = self._today_key(client_ip)
            used = self._counts.get(key, 0)
            if used <= 0:
                return
            self._counts[key] = used - 1
            if self._counts[key] == 0:
                del self._counts[key]
            logger.info("rate_limit_refunded", client_ip=client_ip, remaining=self._limit - (used - 1))


    def _snapshot(self, client_ip: str) -> RateLimitDecision:
        now = datetime.now(UTC)
        key = self._today_key(client_ip, now=now)
        self._prune(keep_day=key[1])
        used = self._counts.get(key, 0)
        remaining = max(0, self._limit - used)
        reset_at = self._reset_at(day=now.date())
        retry_after = max(0, int((reset_at - now).total_seconds()))
        return RateLimitDecision(
            allowed=used < self._limit,
            limit=self._limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after_seconds=retry_after,
        )
