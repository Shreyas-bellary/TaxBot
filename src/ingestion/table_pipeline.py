"""End-to-end table + narrative ingestion pipeline.

Given a parsed :class:`UnstructuredDocument`, this module:

  1. Splits narrative blocks into parent sections and child sentences.
  2. Stores each table's full markdown verbatim as a parent node, and stores
     a Gemini-Flash-generated 3-sentence summary as the embedded child node.
  3. Writes all embeddings (sentence-level + table summary) into
     ``child_nodes`` so the hybrid retriever can match them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
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
from ingestion.embeddings import HuggingFaceEmbedder
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
        *,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._summarizer = summarizer
        self._settings = settings or get_settings()

    async def ingest_document(
        self,
        *,
        record: IRSDocumentRecord,
        document: UnstructuredDocument,
        category: DocCategory,
        language: str = "en",
    ) -> IngestionReport:
        """Persist a freshly parsed document into the parent/child schema."""

        # Replace any prior content for this document so re-ingestion is
        # idempotent. ON DELETE CASCADE removes the matching child rows.
        await self._repo.delete_nodes_for_document(record.doc_id)
        doc_id = await self._repo.upsert_document(
            record,
            category=category,
            language=language,
            extra_metadata={
                "source_url": record.source_url,
            },
        )

        base_metadata = self._base_metadata(record=record, category=category)

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
                        embedding=(),
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
                    embedding=(),
                    metadata={
                        **parent.metadata,
                        "node_kind": "table_summary",
                    },
                )
            )

        # ------------------------------------------------------------------
        # Embed all child texts in one batch
        # ------------------------------------------------------------------
        all_children = [*narrative_children, *table_children]
        if all_children:
            vectors = await self._embedder.embed_batch(
                [child.text_summary for child in all_children]
            )
            embedded_children = [
                child.model_copy(update={"embedding": vector})
                for child, vector in zip(all_children, vectors, strict=True)
            ]
        else:
            embedded_children = []

        # ------------------------------------------------------------------
        # Persist
        # ------------------------------------------------------------------
        all_parents = [*parents_to_insert, *table_parents]
        await self._persist_parents(all_parents)
        children_inserted = await self._repo.insert_children(embedded_children)
        await self._repo.mark_processed(doc_id)

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
    ) -> dict[str, object]:
        return {
            "doc_id": str(record.doc_id),
            "doc_number": record.metadata.doc_number,
            "doc_title": record.metadata.doc_title,
            "doc_type": category.value,
            "form_number": record.metadata.doc_number,
            "tax_year": record.metadata.tax_year,
            "revision_date": record.metadata.revision_date,
            "posted_date": record.metadata.posted_date,
            "source_url": record.source_url,
        }
