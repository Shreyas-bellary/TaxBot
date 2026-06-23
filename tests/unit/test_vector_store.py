"""Unit tests for QdrantVectorStore helper logic.

Network calls are not made; we test the filter-builder and payload helpers
that can be exercised without a live Qdrant cluster.
"""

from __future__ import annotations

from qdrant_client import models

from core.vector_store import _PAYLOAD_INDEXES, _build_metadata_filter, _is_qdrant_upsert_retryable


def test_filter_all_none_returns_none() -> None:
    result = _build_metadata_filter(tax_year=None, doc_type=None, form_numbers=None)
    assert result is None


def test_filter_tax_year_only() -> None:
    result = _build_metadata_filter(tax_year=2024, doc_type=None)
    assert result is not None
    assert len(result.must) == 1  # type: ignore[arg-type]
    cond = result.must[0]  # type: ignore[index]
    assert cond.key == "tax_year"
    assert cond.match.value == 2024  # type: ignore[union-attr]


def test_filter_multiple_conditions() -> None:
    result = _build_metadata_filter(
        tax_year=2023, form_numbers=["Form 1040"], doc_type="form"
    )
    assert result is not None
    assert len(result.must) == 3  # type: ignore[arg-type]
    keys = {c.key for c in result.must}  # type: ignore[union-attr]
    assert keys == {"tax_year", "form_number", "doc_type"}


def test_filter_form_number_only() -> None:
    result = _build_metadata_filter(
        tax_year=None, form_numbers=["Publication 535"], doc_type=None
    )
    assert result is not None
    assert len(result.must) == 1  # type: ignore[arg-type]
    assert result.must[0].key == "form_number"  # type: ignore[index]


def test_payload_indexes_cover_filter_and_delete_fields() -> None:
    indexed = {name for name, _ in _PAYLOAD_INDEXES}
    assert indexed == {"doc_id", "form_number", "doc_type", "tax_year"}
    schema_by_field = dict(_PAYLOAD_INDEXES)
    assert schema_by_field["doc_id"] == models.PayloadSchemaType.KEYWORD
    assert schema_by_field["tax_year"] == models.PayloadSchemaType.INTEGER


def test_qdrant_upsert_retryable_on_write_timeout() -> None:
    import httpx
    from qdrant_client.http.exceptions import ResponseHandlingException

    assert _is_qdrant_upsert_retryable(httpx.WriteTimeout(""))
    wrapped = ResponseHandlingException(source=httpx.WriteTimeout(""))
    assert _is_qdrant_upsert_retryable(wrapped)


def test_qdrant_upsert_not_retryable_on_value_error() -> None:
    assert not _is_qdrant_upsert_retryable(ValueError("bad payload"))
