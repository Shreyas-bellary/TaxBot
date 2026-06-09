"""Tests for the query-filter extractor used by the hybrid retriever."""

from __future__ import annotations

import pytest

from core.retrieval import extract_filters


@pytest.mark.parametrize(
    "query, year, form, doc_type",
    [
        ("What is the 2024 standard deduction for MFJ?", 2024, None, None),
        ("How do I file Form 1040 in 2023?", 2023, "Form 1040", "form"),
        ("Where do I report QBI on Publication 535?", None, "Publication 535", "publication"),
        ("What does Schedule SE cover?", None, "Schedule SE", "form"),
        ("How long can the IRS keep records per the instructions?", None, None, "instruction"),
    ],
)
def test_extract_filters(query: str, year: int | None, form: str | None, doc_type: str | None) -> None:
    filters = extract_filters(query)
    assert filters.tax_year == year
    assert filters.form_number == form
    assert filters.doc_type == doc_type
