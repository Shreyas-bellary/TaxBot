"""Contract tests for the shared Pydantic boundary models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import (
    ChildNode,
    DocCategory,
    IRSDocumentMetadata,
    ParentNode,
)


def test_irs_document_metadata_normalises_url() -> None:
    payload = IRSDocumentMetadata(
        doc_number="Form 1040",
        doc_title="U.S. Individual Income Tax Return",
        revision_date="2024",
        posted_date="01/15/2025",
        pdf_url="https://www.irs.gov/pub/irs-pdf/f1040.pdf",  # type: ignore[arg-type]
    )
    assert str(payload.pdf_url).endswith("/f1040.pdf")
    assert payload.category is DocCategory.FORM
    assert payload.tax_year == 2024


def test_irs_document_metadata_infers_tax_year_from_text() -> None:
    payload = IRSDocumentMetadata(
        doc_number="Publication 17",
        doc_title="Your Federal Income Tax",
        revision_date="Sep 2017",
        posted_date="10/31/2017",
        pdf_url="https://www.irs.gov/pub/irs-pdf/p17.pdf",  # type: ignore[arg-type]
    )
    assert payload.tax_year == 2017
    assert payload.category is DocCategory.PUBLICATION


def test_irs_document_metadata_rejects_unknown_prefix() -> None:
    with pytest.raises(ValidationError):
        IRSDocumentMetadata(
            doc_number="Random Stuff",
            doc_title="x",
            revision_date="2024",
            posted_date="2024",
            pdf_url="https://www.irs.gov/pub/foo.pdf",  # type: ignore[arg-type]
        )


def test_child_node_embedding_dimension_enforced() -> None:
    parent = ParentNode(
        doc_id="00000000-0000-0000-0000-000000000001",  # type: ignore[arg-type]
        text_content="hello",
        metadata={"source_url": "https://www.irs.gov/x"},
    )
    with pytest.raises(ValidationError):
        ChildNode(
            parent_id=parent.id,
            text_summary="short",
            embedding=(0.1, 0.2, 0.3),
            metadata={},
        )


def test_child_node_accepts_empty_or_1024_dim() -> None:
    parent_id = "00000000-0000-0000-0000-000000000002"
    ChildNode(
        parent_id=parent_id,  # type: ignore[arg-type]
        text_summary="ok",
        embedding=(),
        metadata={},
    )
    ChildNode(
        parent_id=parent_id,  # type: ignore[arg-type]
        text_summary="ok",
        embedding=tuple(0.0 for _ in range(1024)),
        metadata={},
    )
