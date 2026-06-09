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
from typing import Any
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
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
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
        """Bulk-insert child nodes. Returns count inserted."""

        payload = list(children)
        if not payload:
            return 0
        records = [
            (
                child.id,
                child.parent_id,
                child.text_summary,
                list(child.embedding),
                _jsonb(child.metadata),
            )
            for child in payload
        ]
        async with self._db.transaction() as connection:
            await connection.executemany(
                """
                INSERT INTO child_nodes (id, parent_id, text_summary, embedding, metadata)
                VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                """,
                records,
            )
        logger.info("children_inserted", count=len(records))
        return len(records)

    # ------------------------------------------------------------------
    # retrieval queries
    # ------------------------------------------------------------------
    async def hybrid_retrieve(
        self,
        *,
        query_text: str,
        query_embedding: Sequence[float],
        top_k_children: int,
        tax_year: int | None,
        form_number: str | None,
        doc_type: str | None,
    ) -> list[dict[str, Any]]:
        """Stage 1 + 2 retrieval.

        Performs vector cosine similarity against ``child_nodes`` with
        metadata pre-filters and an FTS boost, then returns matched child
        rows along with their parent text. The downstream caller maps these
        rows back into typed :class:`RetrievedContext` payloads.
        """

        sql = """
            WITH filtered_children AS (
                SELECT c.id            AS child_id,
                       c.parent_id     AS parent_id,
                       c.text_summary  AS child_summary,
                       c.metadata      AS child_metadata,
                       p.text_content  AS parent_text,
                       p.metadata      AS parent_metadata,
                       p.doc_id        AS doc_id,
                       1 - (c.embedding <=> $1::vector) AS vector_score,
                       ts_rank_cd(
                           to_tsvector('english', c.text_summary),
                           plainto_tsquery('english', $2)
                       ) AS fts_score
                FROM child_nodes c
                JOIN parent_nodes p ON p.id = c.parent_id
                WHERE ($3::int  IS NULL OR (c.metadata ->> 'tax_year')::int  = $3)
                  AND ($4::text IS NULL OR  c.metadata ->> 'form_number'    = $4)
                  AND ($5::text IS NULL OR  c.metadata ->> 'doc_type'       = $5)
            )
            SELECT *,
                   (0.75 * vector_score) + (0.25 * fts_score) AS hybrid_score
            FROM filtered_children
            ORDER BY hybrid_score DESC
            LIMIT $6
        """
        rows = await self._db.fetch(
            sql,
            list(query_embedding),
            query_text,
            tax_year,
            form_number,
            doc_type,
            top_k_children,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            record["child_metadata"] = _coerce_metadata(record["child_metadata"])
            record["parent_metadata"] = _coerce_metadata(record["parent_metadata"])
            result.append(record)
        return result
