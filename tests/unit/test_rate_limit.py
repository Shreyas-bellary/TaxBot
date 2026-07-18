"""Unit tests for the in-memory IP daily rate limiter."""

from __future__ import annotations

import pytest

from core.rate_limit import IpDailyRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_up_to_limit() -> None:
    limiter = IpDailyRateLimiter(limit=3)
    ip = "203.0.113.10"

    for remaining_after in (2, 1, 0):
        decision = await limiter.consume(ip)
        assert decision.allowed is True
        assert decision.remaining == remaining_after

    blocked = await limiter.consume(ip)
    assert blocked.allowed is False
    assert blocked.remaining == 0
    assert blocked.retry_after_seconds > 0


@pytest.mark.asyncio
async def test_rate_limiter_isolates_ips() -> None:
    limiter = IpDailyRateLimiter(limit=1)
    first = await limiter.consume("198.51.100.1")
    second = await limiter.consume("198.51.100.2")
    assert first.allowed is True
    assert second.allowed is True
    assert (await limiter.consume("198.51.100.1")).allowed is False


@pytest.mark.asyncio
async def test_rate_limiter_refund() -> None:
    limiter = IpDailyRateLimiter(limit=1)
    ip = "192.0.2.55"
    reserved = await limiter.consume(ip)
    assert reserved.allowed is True
    await limiter.refund(ip)
    again = await limiter.consume(ip)
    assert again.allowed is True
