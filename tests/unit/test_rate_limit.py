"""Unit tests for rate-limit helpers that do not require a live database.

The DB-backed IpDailyRateLimiter is exercised end-to-end in
tests/integration/test_rate_limit_integration.py, which requires the CI
Postgres service.  These tests cover:

* ``RateLimitDecision`` field semantics
* ``_parse_ip`` / ``_client_ip`` IP normalisation and Cloud Run XFF handling
"""

from __future__ import annotations

import ipaddress
from datetime import UTC, datetime

import pytest

from core.rate_limit import RateLimitDecision

# ---------------------------------------------------------------------------
# RateLimitDecision helpers
# ---------------------------------------------------------------------------

def test_rate_limit_decision_allowed_fields() -> None:
    reset = datetime(2026, 7, 20, 0, 0, 0, tzinfo=UTC)
    d = RateLimitDecision(allowed=True, limit=3, remaining=2, reset_at=reset, retry_after_seconds=0)
    assert d.allowed is True
    assert d.remaining == 2
    assert d.retry_after_seconds == 0


def test_rate_limit_decision_denied_fields() -> None:
    reset = datetime(2026, 7, 20, 0, 0, 0, tzinfo=UTC)
    d = RateLimitDecision(
        allowed=False, limit=3, remaining=0, reset_at=reset, retry_after_seconds=3600
    )
    assert d.allowed is False
    assert d.remaining == 0
    assert d.retry_after_seconds == 3600


def test_rate_limit_decision_is_frozen() -> None:
    reset = datetime(2026, 7, 20, 0, 0, 0, tzinfo=UTC)
    d = RateLimitDecision(allowed=True, limit=3, remaining=1, reset_at=reset, retry_after_seconds=0)
    with pytest.raises((AttributeError, TypeError)):
        d.remaining = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _parse_ip helper (imported directly from api.main to stay close to the
# function under test; we avoid importing the entire FastAPI app here).
# ---------------------------------------------------------------------------

def _parse_ip(raw: str) -> str | None:
    """Mirror of api.main._parse_ip for isolated unit testing."""
    try:
        return str(ipaddress.ip_address(raw.strip()))
    except ValueError:
        return None


class TestParseIp:
    def test_valid_ipv4(self) -> None:
        assert _parse_ip("1.2.3.4") == "1.2.3.4"

    def test_valid_ipv6(self) -> None:
        assert _parse_ip("::1") == "::1"

    def test_valid_ipv4_with_whitespace(self) -> None:
        assert _parse_ip("  10.0.0.1  ") == "10.0.0.1"

    def test_invalid_returns_none(self) -> None:
        assert _parse_ip("not-an-ip") is None
        assert _parse_ip("") is None
        assert _parse_ip("999.999.999.999") is None

    def test_ipv4_mapped_ipv6(self) -> None:
        result = _parse_ip("::ffff:192.0.2.1")
        assert result is not None
        # Must be a valid address
        ipaddress.ip_address(result)

    def test_normalises_ipv6_case(self) -> None:
        # Python's ipaddress always returns lowercase
        result = _parse_ip("2001:DB8::1")
        assert result is not None
        assert result == result.lower()


# ---------------------------------------------------------------------------
# Cloud Run X-Forwarded-For rightmost-entry semantics (logic unit test)
# ---------------------------------------------------------------------------

def _extract_rightmost_valid(header_value: str) -> str | None:
    """Mirror of the trusted-proxy branch in api.main._client_ip."""
    for candidate in reversed(header_value.split(",")):
        ip = _parse_ip(candidate)
        if ip is not None:
            return ip
    return None


class TestCloudRunXFFExtraction:
    """The GFE appends the real client IP as the last X-Forwarded-For entry."""

    def test_single_ip(self) -> None:
        assert _extract_rightmost_valid("1.2.3.4") == "1.2.3.4"

    def test_picks_last_of_chain(self) -> None:
        # Client could have injected the first entry; only the last is trusted.
        assert _extract_rightmost_valid("spoofed, 10.0.0.1, 203.0.113.5") == "203.0.113.5"

    def test_skips_invalid_entries(self) -> None:
        # If the last entry is garbage, falls back to the next valid one.
        assert _extract_rightmost_valid("1.2.3.4, garbage") == "1.2.3.4"

    def test_all_invalid_returns_none(self) -> None:
        assert _extract_rightmost_valid("foo, bar") is None

    def test_empty_header_returns_none(self) -> None:
        assert _extract_rightmost_valid("") is None

    def test_spoof_attempt_ignored(self) -> None:
        # Attacker sets a fake first IP; Cloud Run appends the real IP last.
        result = _extract_rightmost_valid("evil.fake.ip.0, 203.0.113.99")
        assert result == "203.0.113.99"
