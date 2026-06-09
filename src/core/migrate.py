"""Minimal idempotent SQL migrator.

Applies every ``*.sql`` file inside :mod:`core.migrations` in lexicographic
order. Each migration runs inside a single transaction and is recorded in
``schema_migrations`` so that re-runs become no-ops.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.config import get_settings
from core.db import Database
from core.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _migration_files() -> list[Path]:
    """Return every migration shipped inside ``core/migrations``."""

    if not _MIGRATIONS_DIR.is_dir():
        raise FileNotFoundError(f"Migrations directory missing: {_MIGRATIONS_DIR}")
    return sorted(_MIGRATIONS_DIR.glob("*.sql"))


async def apply_migrations(database: Database) -> list[str]:
    """Apply every pending migration. Returns the names that were applied."""

    await database.execute(_TRACKING_TABLE_SQL)

    applied: list[str] = []
    for migration_path in _migration_files():
        name = migration_path.name
        row = await database.fetchrow(
            "SELECT filename FROM schema_migrations WHERE filename = $1",
            name,
        )
        if row is not None:
            logger.info("migration_skipped", name=name)
            continue

        sql = migration_path.read_text(encoding="utf-8")
        async with database.transaction() as connection:
            await connection.execute(sql)
            await connection.execute(
                "INSERT INTO schema_migrations (filename) VALUES ($1)",
                name,
            )
        logger.info("migration_applied", name=name)
        applied.append(name)

    return applied


async def _run() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)
    async with Database(settings) as database:
        applied = await apply_migrations(database)
    logger.info("migrations_complete", count=len(applied), names=applied)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
