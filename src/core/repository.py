"""Repository layer for TaxBot persistence.

Every SQL operation against ``ingested_documents``, ``parent_nodes`` and
``child_nodes`` flows through this module. Higher layers (ingestion,
retrieval) only ever speak in terms of the Pydantic models from
:mod:`core.models` and the typed methods below.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from core.db import Database
from core.logging_config import get_logger
from core.models import (
    ChildNode,
    DocCategory,
    IRSDocumentRecord,
    ParentNode,
)

logger = get_logger(__name__)


def _coerce_metadata(value: Any) -> dict[str, object]:
    """Decode a JSONB value (asyncpg returns either dict or str)."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes | bytearray):
        return cast(dict[str, object], json.loads(value.decode("utf-8")))
    if isinstance(value, str):
        return cast(dict[str, object], json.loads(value))
    raise TypeError(f"Unsupported metadata payload: {type(value)!r}")


def _jsonb(value: dict[str, object]) -> str:
    """Serialise a Python dict for an asyncpg JSONB column."""

    return json.dumps(value, separators=(",", ":"), default=str)


class DocumentRepository:
    """All persistence operations the ingestion pipeline needs."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------
    # ingested_documents
    # ------------------------------------------------------------------
    async def get_existing_document(self, pdf_url: str) -> dict[str, Any] | None:
        row = await self._db.fetchrow(
            """
            SELECT doc_id, doc_number, doc_title, pdf_url, revision_date,
                   posted_date, tax_year, category, language, pdf_sha256,
                   metadata, last_seen_at, processed_at
            FROM ingested_documents
            WHERE pdf_url = $1
            """,
            pdf_url,
        )
        if row is None:
            return None
        record = dict(row)
        record["metadata"] = _coerce_metadata(record["metadata"])
        return record

    async def upsert_document(
        self,
        record: IRSDocumentRecord,
        *,
        category: DocCategory,
        language: str,
        extra_metadata: dict[str, object] | None = None,
    ) -> UUID:
        """Insert or update an ingestion record and return its UUID."""

        metadata = dict(extra_metadata or {})
        row = await self._db.fetchrow(
            """
            INSERT INTO ingested_documents (
                doc_id, doc_number, doc_title, pdf_url, revision_date,
                posted_date, tax_year, category, language, pdf_sha256, metadata,
                last_seen_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, now()
            )
            ON CONFLICT (pdf_url) DO UPDATE SET
                doc_number    = EXCLUDED.doc_number,
                doc_title     = EXCLUDED.doc_title,
                revision_date = EXCLUDED.revision_date,
                posted_date   = EXCLUDED.posted_date,
                tax_year      = EXCLUDED.tax_year,
                category      = EXCLUDED.category,
                language      = EXCLUDED.language,
                pdf_sha256    = COALESCE(EXCLUDED.pdf_sha256, ingested_documents.pdf_sha256),
                metadata      = ingested_documents.metadata || EXCLUDED.metadata,
                last_seen_at  = now()
            RETURNING doc_id
            """,
            record.doc_id,
            record.metadata.doc_number,
            record.metadata.doc_title,
            str(record.metadata.pdf_url),
            record.metadata.revision_date,
            record.metadata.posted_date,
            record.metadata.tax_year,
            category.value,
            language,
            record.pdf_sha256,
            _jsonb(metadata),
        )
        assert row is not None
        return UUID(str(row["doc_id"]))

    async def mark_processed(self, doc_id: UUID, *, when: datetime | None = None) -> None:
        await self._db.execute(
            """
            UPDATE ingested_documents
            SET processed_at = COALESCE($2, now())
            WHERE doc_id = $1
            """,
            doc_id,
            when,
        )

    async def list_processed_urls(self) -> set[str]:
        rows = await self._db.fetch(
            "SELECT pdf_url FROM ingested_documents WHERE processed_at IS NOT NULL"
        )
        return {row["pdf_url"] for row in rows}

    # ------------------------------------------------------------------
    # parent_nodes
    # ------------------------------------------------------------------
    async def delete_nodes_for_document(self, doc_id: UUID) -> None:
        """Delete all parent + child nodes attached to a document (cascade)."""

        await self._db.execute("DELETE FROM parent_nodes WHERE doc_id = $1", doc_id)

    async def insert_parent(self, parent: ParentNode) -> UUID:
        row = await self._db.fetchrow(
            """
            INSERT INTO parent_nodes (id, doc_id, text_content, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id
            """,
            parent.id,
            parent.doc_id,
            parent.text_content,
            _jsonb(parent.metadata),
        )
        assert row is not None
        return UUID(str(row["id"]))

    async def insert_children(self, children: Iterable[ChildNode]) -> int:
        """Bulk-insert child nodes (text + metadata only). Returns count inserted.
        """
        payload = list(children)
        if not payload:
            return 0
        records = [
            (
                child.id,
                child.parent_id,
                child.text_summary,
                _jsonb(child.metadata),
            )
            for child in payload
        ]
        async with self._db.transaction() as connection:
            await connection.executemany(
                """
                INSERT INTO child_nodes (id, parent_id, text_summary, metadata)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                records,
            )
        logger.info("children_inserted", count=len(records))
        return len(records)

    # ------------------------------------------------------------------
    # retrieval queries
    # ------------------------------------------------------------------

    async def fetch_parents(
        self, parent_ids: Sequence[UUID]
    ) -> dict[UUID, dict[str, Any]]:
        """Fetch parent node rows for the given IDs in one round-trip.

        Returns a mapping from ``parent_id`` → row dict with keys
        ``doc_id``, ``text_content``, and ``metadata`` (decoded JSONB).
        IDs not found in the database are omitted from the result.
        """
        if not parent_ids:
            return {}
        rows = await self._db.fetch(
            """
            SELECT id, doc_id, text_content, metadata
            FROM parent_nodes
            WHERE id = ANY($1::uuid[])
            """,
            [str(pid) for pid in parent_ids],
        )
        result: dict[UUID, dict[str, Any]] = {}
        for row in rows:
            record = dict(row)
            record["metadata"] = _coerce_metadata(record["metadata"])
            result[UUID(str(record["id"]))] = record
        return result
