"""Historical backfill entry point.

Walks the IRS AJAX listing once (last 5 tax years), downloads each PDF,
parses it via Unstructured, and persists parent/child nodes through the
table pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import NoReturn

from core.config import Settings, get_settings
from core.db import Database
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


async def run_backfill(
    *,
    settings: Settings | None = None,
    max_documents: int | None = None,
    concurrency: int = 4,
) -> dict[str, int]:
    """Execute the historical backfill. Returns aggregate counts."""

    settings = settings or get_settings()
    counts = {
        "scraped": 0,
        "filtered_window": 0,
        "filtered_existing": 0,
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

        try:
            processed_urls = await repository.list_processed_urls()
            semaphore = asyncio.Semaphore(concurrency)
            tasks: list[asyncio.Task[None]] = []

            async def _worker(metadata: IRSDocumentMetadata) -> None:
                async with semaphore:
                    try:
                        await _ingest_one(
                            metadata=metadata,
                            repository=repository,
                            pdf_fetcher=pdf_fetcher,
                            unstructured=unstructured,
                            pipeline=pipeline,
                        )
                        counts["ingested"] += 1
                    except Exception as exc:
                        counts["failed"] += 1
                        logger.error(
                            "backfill_document_failed",
                            doc_number=metadata.doc_number,
                            pdf_url=str(metadata.pdf_url),
                            error=str(exc),
                        )

            async for metadata in _scrape_window(scraper, settings, counts):
                if str(metadata.pdf_url) in processed_urls:
                    counts["filtered_existing"] += 1
                    continue
                if max_documents is not None and counts["ingested"] >= max_documents:
                    break
                tasks.append(asyncio.create_task(_worker(metadata)))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=False)
        finally:
            await vector_store.aclose()

    logger.info("backfill_complete", **counts)
    return counts


async def _scrape_window(
    scraper: IRSAJAXClient,
    settings: Settings,
    counts: dict[str, int],
) -> AsyncIterator[IRSDocumentMetadata]:
    async for metadata in scraper.iter_documents(sort="natural", drop_multilingual=True):
        counts["scraped"] += 1
        if not is_within_backfill_window(metadata, settings=settings):
            counts["filtered_window"] += 1
            continue
        yield metadata


async def _ingest_one(
    *,
    metadata: IRSDocumentMetadata,
    repository: DocumentRepository,
    pdf_fetcher: PDFFetcher,
    unstructured: UnstructuredParser,
    pipeline: TablePipeline,
) -> None:
    fetched = await pdf_fetcher.fetch(str(metadata.pdf_url))
    record = IRSDocumentRecord(
        metadata=metadata,
        pdf_sha256=fetched.sha256,
        fetched_at=datetime.now(tz=UTC),
    )
    document = await unstructured.partition(
        filename=_filename_from_url(str(metadata.pdf_url)),
        content=fetched.content,
    )
    await pipeline.ingest_document(
        record=record,
        document=document,
        category=metadata.category,
    )
    # Best-effort persistence of the document hash for future delta runs.
    await repository.upsert_document(
        record,
        category=metadata.category,
        language="en",
        extra_metadata={"source_url": record.source_url},
    )


def _filename_from_url(url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or "irs_document.pdf"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TaxBot historical backfill")
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional hard cap on the number of documents to ingest.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max simultaneous document ingestions.",
    )
    return parser.parse_args()


def main() -> NoReturn:
    args = _parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)
    asyncio.run(
        run_backfill(
            settings=settings,
            max_documents=args.max_documents,
            concurrency=args.concurrency,
        )
    )
    raise SystemExit(0)


if __name__ == "__main__":
    main()
