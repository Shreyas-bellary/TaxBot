"""Nightly delta updater.

Runs on a GitHub Actions cron at 00:00 UTC. The behaviour is identical to
the backfill except that:

  * The scraper sorts by ``posted_date_desc`` so the most recently posted
    documents appear first. As soon as we encounter records that already
    match Supabase state on both metadata and PDF hash, we short-circuit.
  * We reprocess a document if **either** its listing metadata changed
    (revision/posted date, title, category) **or** the downloaded PDF
    SHA-256 differs from the previously stored hash.
  * Documents that have not been seen in the latest scrape have their
    ``last_seen_at`` left untouched, so analytics can audit drift.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from typing import Any, NoReturn

from core.config import Settings, get_settings
from core.db import Database
from core.errors import UnsupportedPDFError
from core.logging_config import configure_logging, get_logger
from core.models import IRSDocumentMetadata, IRSDocumentRecord
from core.repository import DocumentRepository
from core.vector_store import QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder
from ingestion.filters import is_within_backfill_window
from ingestion.irs_scraper import IRSAJAXClient
from ingestion.pdf_fetcher import PDFFetcher
from ingestion.sparse_encoder import SparseEncoder
from ingestion.summarizer import TableSummarizer
from ingestion.table_pipeline import TablePipeline
from ingestion.unstructured_parser import UnstructuredParser

logger = get_logger(__name__)

# Number of consecutive "unchanged" rows that triggers an early exit when
# scraping in posted-date-desc order.
_UNCHANGED_STREAK_LIMIT = 50


async def run_delta(
    *,
    settings: Settings | None = None,
    max_pages: int | None = None,
    concurrency: int = 2,
) -> dict[str, int]:
    """Execute one delta sweep against the IRS listing."""

    settings = settings or get_settings()
    counts = {
        "scraped": 0,
        "filtered_window": 0,
        "unchanged": 0,
        "metadata_changed": 0,
        "hash_changed": 0,
        "new_documents": 0,
        "filtered_unsupported": 0,
        "ingested": 0,
        "failed": 0,
    }

    async with (
        Database(settings) as database,
        PDFFetcher(settings) as pdf_fetcher,
        HuggingFaceEmbedder(settings) as embedder,
        IRSAJAXClient(settings) as scraper,
    ):
        repository = DocumentRepository(database)
        unstructured = UnstructuredParser(settings)
        summarizer = TableSummarizer(settings)
        vector_store = QdrantVectorStore(settings)
        await vector_store.ensure_collection()
        sparse_encoder = SparseEncoder(settings)
        pipeline = TablePipeline(
            repository,
            embedder,
            summarizer,
            vector_store,
            sparse_encoder,
            settings=settings,
        )

        unchanged_streak = 0
        try:
            async for metadata in scraper.iter_documents(
                sort="posted_date_desc",
                drop_multilingual=True,
                max_pages=max_pages,
            ):
                counts["scraped"] += 1
                if not is_within_backfill_window(metadata, settings=settings):
                    counts["filtered_window"] += 1
                    continue

                existing = await repository.get_existing_document(str(metadata.pdf_url))
                metadata_diff = _metadata_changed(existing, metadata)

                if existing is not None and not metadata_diff:
                    # Need to verify PDF hash before we can declare "unchanged".
                    try:
                        fetched = await pdf_fetcher.fetch(str(metadata.pdf_url))
                    except UnsupportedPDFError as exc:
                        counts["filtered_unsupported"] += 1
                        logger.warning(
                            "delta_pdf_unsupported_skipped",
                            doc_number=metadata.doc_number,
                            pdf_url=str(metadata.pdf_url),
                            reason=str(exc),
                        )
                        continue
                    if existing.get("pdf_sha256") == fetched.sha256:
                        counts["unchanged"] += 1
                        unchanged_streak += 1
                        await _touch_last_seen(repository, metadata)
                        if unchanged_streak >= _UNCHANGED_STREAK_LIMIT:
                            logger.info(
                                "delta_unchanged_streak_short_circuit",
                                streak=unchanged_streak,
                            )
                            break
                        continue
                    counts["hash_changed"] += 1
                    await _reingest(
                        metadata=metadata,
                        fetched_content=fetched.content,
                        fetched_sha=fetched.sha256,
                        repository=repository,
                        unstructured=unstructured,
                        pipeline=pipeline,
                    )
                    counts["ingested"] += 1
                    unchanged_streak = 0
                    continue

                unchanged_streak = 0
                if existing is None:
                    counts["new_documents"] += 1
                else:
                    counts["metadata_changed"] += 1

                try:
                    fetched = await pdf_fetcher.fetch(str(metadata.pdf_url))
                    await _reingest(
                        metadata=metadata,
                        fetched_content=fetched.content,
                        fetched_sha=fetched.sha256,
                        repository=repository,
                        unstructured=unstructured,
                        pipeline=pipeline,
                    )
                    counts["ingested"] += 1
                except UnsupportedPDFError as exc:
                    counts["filtered_unsupported"] += 1
                    logger.warning(
                        "delta_pdf_unsupported_skipped",
                        doc_number=metadata.doc_number,
                        pdf_url=str(metadata.pdf_url),
                        reason=str(exc),
                    )
                except Exception as exc:
                    counts["failed"] += 1
                    logger.error(
                        "delta_document_failed",
                        doc_number=metadata.doc_number,
                        pdf_url=str(metadata.pdf_url),
                        error=str(exc),
                    )
        finally:
            await vector_store.aclose()

    logger.info("delta_complete", **counts)
    return counts


async def _touch_last_seen(
    repository: DocumentRepository,
    metadata: IRSDocumentMetadata,
) -> None:
    record = IRSDocumentRecord(metadata=metadata, fetched_at=datetime.now(tz=UTC))
    await repository.upsert_document(
        record,
        category=metadata.category,
        language="en",
    )


def _metadata_changed(
    existing: dict[str, Any] | None,
    fresh: IRSDocumentMetadata,
) -> bool:
    if existing is None:
        return True
    return (
        existing.get("doc_number") != fresh.doc_number
        or existing.get("doc_title") != fresh.doc_title
        or existing.get("revision_date") != fresh.revision_date
        or existing.get("posted_date") != fresh.posted_date
        or existing.get("category") != fresh.category.value
    )


async def _reingest(
    *,
    metadata: IRSDocumentMetadata,
    fetched_content: bytes,
    fetched_sha: str,
    repository: DocumentRepository,
    unstructured: UnstructuredParser,
    pipeline: TablePipeline,
) -> None:
    record = IRSDocumentRecord(
        metadata=metadata,
        pdf_sha256=fetched_sha,
        fetched_at=datetime.now(tz=UTC),
    )
    document = await unstructured.partition(
        filename=_filename_from_url(str(metadata.pdf_url)),
        content=fetched_content,
    )
    await pipeline.ingest_document(
        record=record,
        document=document,
        category=metadata.category,
    )
    await repository.upsert_document(
        record,
        category=metadata.category,
        language="en",
    )


def _filename_from_url(url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or "irs_document.pdf"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TaxBot nightly delta update")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional hard limit on number of AJAX pages to walk.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max simultaneous document ingestions.",
    )
    return parser.parse_args()


def main() -> NoReturn:
    args = _parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)
    asyncio.run(
        run_delta(
            settings=settings,
            max_pages=args.max_pages,
            concurrency=args.concurrency,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
