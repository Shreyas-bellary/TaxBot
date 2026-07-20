"""Tests for migration-job configuration."""

from __future__ import annotations

from core.migrate import MigrationSettings


def test_migration_settings_require_only_postgres() -> None:
    settings = MigrationSettings(
        postgres_dsn="postgresql://taxbot:taxbot@localhost:5432/taxbot",
        _env_file=None,
    )

    assert str(settings.postgres_dsn).startswith("postgresql://taxbot:taxbot@localhost")
    assert settings.db_command_timeout_seconds == 30.0
