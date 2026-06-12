"""End-to-end table + narrative ingestion pipeline.

Given a parsed :class:`UnstructuredDocument`, this module:

  1. Splits narrative blocks into parent sections and child sentences.
  2. Stores each table's full markdown verbatim as a parent node, and stores
     a Gemini-Flash-generated 3-sentence summary as the embedded child node.
  3. Writes dense embeddings and BM25 sparse vectors into Qdrant; writes
     text-only child rows into Postgres ``child_nodes``.

Deletion and consistency contract
----------------------------------
Re-ingestion of an existing document follows a delete-before-insert pattern
to keep Postgres and Qdrant consistent:

1. ``upsert_document`` resolves the canonical ``doc_id`` (via ``ON CONFLICT``
   on ``pdf_url``).
2. ``delete_nodes_for_document(doc_id)`` removes Postgres parent rows;
   ``child_nodes`` are removed via ``ON DELETE CASCADE``.
3. ``vector_store.delete_by_doc_id(doc_id)`` removes all matching Qdrant
   points.
4. New parents + children are written to Postgres, then upserted to Qdrant.
5. **Rollback on Qdrant failure**: if the Qdrant upsert fails after Postgres
   writes succeed, the pipeline deletes the freshly written Postgres rows
   (``delete_nodes_for_document``) and the Qdrant points (best-effort), then
   re-raises the exception.  This leaves both stores in the pre-ingest state,
   which is equivalent to a failed fresh ingest for a backfill retry.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from core.config import Settings, get_settings
from core.logging_config import get_logger
from core.models import (
    ChildNode,
    DocCategory,
    IRSDocumentRecord,
    ParentNode,
)
from core.repository import DocumentRepository
from core.vector_store import QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder
from ingestion.sparse_encoder import SparseEncoder
from ingestion.summarizer import TableSummarizer, TableSummaryInput
from ingestion.text_splitter import (
    group_narratives_into_parents,
    split_into_child_sentences,
)
from ingestion.unstructured_parser import UnstructuredDocument

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Counts emitted by a single document ingestion."""

    doc_id: UUID
    parents_inserted: int
    children_inserted: int
    tables_summarised: int


class TablePipeline:
    """Combine parsing, summarisation, embedding, and persistence."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedder: HuggingFaceEmbedder,
        summarizer: TableSummarizer,
        vector_store: QdrantVectorStore,
        sparse_encoder: SparseEncoder,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._summarizer = summarizer
        self._vector_store = vector_store
        self._sparse_encoder = sparse_encoder
        self._settings = settings or get_settings()

    async def ingest_document(
        self,
        *,
        record: IRSDocumentRecord,
        document: UnstructuredDocument,
        category: DocCategory,
        language: str = "en",
    ) -> IngestionReport:
        """Persist a freshly parsed document into the parent/child schema.

        See module docstring for the deletion and consistency contract.
        """
        doc_id = await self._repo.upsert_document(
            record,
            category=category,
            language=language,
            extra_metadata={
                "source_url": record.source_url,
            },
        )
        await self._repo.delete_nodes_for_document(doc_id)
        await self._vector_store.delete_by_doc_id(doc_id)

        base_metadata = self._base_metadata(
            record=record,
            category=category,
            canonical_doc_id=doc_id,
        )

        # ------------------------------------------------------------------
        # Narrative parents and sentence children
        # ------------------------------------------------------------------
        parent_blocks = group_narratives_into_parents(document.narratives)
        narrative_children: list[ChildNode] = []
        parents_to_insert: list[ParentNode] = []

        for parent_text in parent_blocks:
            parent = ParentNode(
                doc_id=doc_id,
                text_content=parent_text,
                metadata={**base_metadata, "node_kind": "narrative"},
            )
            parents_to_insert.append(parent)
            for sentence_chunk in split_into_child_sentences(parent_text):
                narrative_children.append(
                    ChildNode(
                        parent_id=parent.id,
                        text_summary=sentence_chunk,
                        metadata={**base_metadata, "node_kind": "sentence"},
                    )
                )

        # ------------------------------------------------------------------
        # Table parents (full markdown) + table summary children
        # ------------------------------------------------------------------
        table_summary_inputs: list[TableSummaryInput] = []
        table_parents: list[ParentNode] = []
        for table in document.tables:
            if not table.markdown.strip():
                continue
            parent = ParentNode(
                doc_id=doc_id,
                text_content=table.markdown,
                metadata={
                    **base_metadata,
                    "node_kind": "table",
                    "page_number": table.page_number,
                    "section": table.section,
                },
            )
            table_parents.append(parent)
            table_summary_inputs.append(
                TableSummaryInput(
                    doc_number=record.metadata.doc_number,
                    doc_title=record.metadata.doc_title,
                    tax_year=record.metadata.tax_year,
                    table_markdown=table.markdown,
                )
            )

        table_summaries = await asyncio.gather(
            *(self._summarizer.summarize(payload) for payload in table_summary_inputs),
            return_exceptions=True,
        )

        table_children: list[ChildNode] = []
        for parent, summary_result in zip(table_parents, table_summaries, strict=True):
            if isinstance(summary_result, BaseException):
                logger.warning(
                    "table_summary_failed",
                    parent_id=str(parent.id),
                    error=str(summary_result),
                )
                continue
            table_children.append(
                ChildNode(
                    parent_id=parent.id,
                    text_summary=summary_result,
                    metadata={
                        **parent.metadata,
                        "node_kind": "table_summary",
                    },
                )
            )

        # ------------------------------------------------------------------
        # Encode: dense embeddings (HF) + sparse BM25 (fastembed)
        # ------------------------------------------------------------------
        all_children = [*narrative_children, *table_children]
        if all_children:
            summaries = [child.text_summary for child in all_children]
            dense_vectors, sparse_vectors = await asyncio.gather(
                self._embedder.embed_batch(summaries),
                self._sparse_encoder.embed_documents(summaries),
            )
        else:
            dense_vectors = []
            sparse_vectors = []

        # ------------------------------------------------------------------
        # Persist to Postgres
        # ------------------------------------------------------------------
        all_parents = [*parents_to_insert, *table_parents]
        await self._persist_parents(all_parents)
        children_inserted = await self._repo.insert_children(all_children)
        await self._repo.mark_processed(doc_id)

        if all_children:
            try:
                await self._vector_store.upsert_points(
                    child_ids=[str(child.id) for child in all_children],
                    dense_vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                    payloads=[_qdrant_payload(child) for child in all_children],
                    doc_id=str(doc_id),
                )
            except Exception as exc:
                logger.error(
                    "qdrant_upsert_failed_rolling_back",
                    doc_id=str(doc_id),
                    error=str(exc),
                )
                await self._repo.delete_nodes_for_document(doc_id)
                with contextlib.suppress(Exception):
                    await self._vector_store.delete_by_doc_id(doc_id)
                raise

        report = IngestionReport(
            doc_id=doc_id,
            parents_inserted=len(all_parents),
            children_inserted=children_inserted,
            tables_summarised=len(table_children),
        )
        logger.info(
            "document_ingested",
            doc_id=str(report.doc_id),
            doc_number=record.metadata.doc_number,
            parents=report.parents_inserted,
            children=report.children_inserted,
            tables=report.tables_summarised,
        )
        return report

    async def _persist_parents(self, parents: Iterable[ParentNode]) -> None:
        for parent in parents:
            await self._repo.insert_parent(parent)

    @staticmethod
    def _base_metadata(
        *,
        record: IRSDocumentRecord,
        category: DocCategory,
        canonical_doc_id: UUID,
    ) -> dict[str, object]:
        return {
            "doc_id": str(canonical_doc_id),
            "doc_number": record.metadata.doc_number,
            "doc_title": record.metadata.doc_title,
            "doc_type": category.value,
            "form_number": record.metadata.doc_number,
            "tax_year": record.metadata.tax_year,
            "revision_date": record.metadata.revision_date,
            "posted_date": record.metadata.posted_date,
            "source_url": record.source_url,
        }


def _qdrant_payload(child: ChildNode) -> dict[str, Any]:
    """Build the Qdrant point payload from a child node's metadata.

    Only the fields used by retrieval filters and cross-reference lookups are
    included in the payload; the full metadata is in Postgres.
    """
    meta = child.metadata
    return {
        "parent_id": str(child.parent_id),
        "doc_id": str(meta.get("doc_id", "")),
        "tax_year": meta.get("tax_year"),
        "form_number": meta.get("form_number"),
        "doc_type": meta.get("doc_type"),
        "node_kind": meta.get("node_kind"),
    }
