"""Tests for ingestion filter predicates."""

from __future__ import annotations

from io import BytesIO

import pytest
from pydantic import SecretStr
from pypdf import PdfWriter

from core.config import Settings
from core.errors import OversizedPublicationError
from core.models import IRSDocumentMetadata
from ingestion.filters import is_within_backfill_window, reject_oversized_publication


def _settings(**overrides: object) -> Settings:
    base = {
        "postgres_dsn": "postgresql://postgres:postgres@localhost:5432/postgres",
        "unstructured_api_key": SecretStr("key"),
        "huggingface_api_token": SecretStr("token"),
        "gemini_api_key": SecretStr("gemini"),
        "qdrant_url": "https://test.qdrant.io:6333",
        "qdrant_api_key": SecretStr("qdrant"),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _metadata(doc_number: str = "Publication 17") -> IRSDocumentMetadata:
    return IRSDocumentMetadata(
        doc_number=doc_number,
        doc_title="Your Federal Income Tax",
        revision_date="2024",
        posted_date="01/01/2024",
        pdf_url="https://www.irs.gov/pub/irs-pdf/p17.pdf",  # type: ignore[arg-type]
    )


def _pdf_bytes(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_reject_oversized_publication_skips_when_over_limit() -> None:
    settings = _settings(publication_max_pages=200)
    with pytest.raises(OversizedPublicationError, match="has 201 pages"):
        reject_oversized_publication(
            _metadata(),
            _pdf_bytes(201),
            settings=settings,
        )


def test_reject_oversized_publication_allows_at_limit() -> None:
    settings = _settings(publication_max_pages=200)
    reject_oversized_publication(_metadata(), _pdf_bytes(200), settings=settings)


def test_reject_oversized_publication_disabled_when_zero() -> None:
    settings = _settings(publication_max_pages=0)
    reject_oversized_publication(_metadata(), _pdf_bytes(500), settings=settings)


def test_reject_oversized_publication_ignores_non_publications() -> None:
    settings = _settings(publication_max_pages=10)
    metadata = _metadata(doc_number="Form 1040")
    reject_oversized_publication(metadata, _pdf_bytes(500), settings=settings)


def test_is_within_backfill_window_respects_tax_year() -> None:
    settings = _settings(backfill_oldest_tax_year=2023)
    assert is_within_backfill_window(_metadata(), settings=settings) is True
