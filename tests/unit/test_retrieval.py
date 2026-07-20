"""Tests for the Qdrant-backed hybrid retrieval pipeline.

All external I/O (Qdrant, Postgres, HF embeddings, sparse encoder) is
replaced with lightweight fakes so these run without any live services.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from core.config import get_settings
from core.errors import OutOfDomainQueryError, RetrievalError
from core.query_router import QueryRouteResult, RouteFilters
from core.retrieval import (
    HybridRetriever,
    _normalize_rows,
    assess_retrieval_confidence,
)
from core.vector_store import HybridSearchResult

# ---------------------------------------------------------------------------
# Auto-patch the query router for all retrieve() tests in this module.
# This avoids real LLM API calls; individual tests can override as needed.
# ---------------------------------------------------------------------------

_DEFAULT_ROUTE = QueryRouteResult(
    filters=RouteFilters(tax_year=None, doc_type=None)
)

# A route with a year filter so relaxation tests can trigger the retry path
_FILTERED_ROUTE = QueryRouteResult(
    filters=RouteFilters(tax_year=2024, doc_type=None)
)


@pytest.fixture(autouse=True)
def _patch_router(monkeypatch: pytest.MonkeyPatch):  # type: ignore[return]
    """Patch route_query to return an empty-filter route by default."""
    with patch(
        "core.retrieval.route_query",
        new=AsyncMock(return_value=_DEFAULT_ROUTE),
    ) as mock:
        yield mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARENT_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_CHILD_ID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_DOC_ID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _make_hit(
    rrf_score: float = 0.5,
    dense_top_score: float = 0.8,
    parent_id: str | None = None,
    child_id: str | None = None,
    doc_id: str | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        child_id=child_id or str(_CHILD_ID),
        parent_id=parent_id or str(_PARENT_ID),
        doc_id=doc_id or str(_DOC_ID),
        rrf_score=rrf_score,
        dense_top_score=dense_top_score,
    )


# ---------------------------------------------------------------------------
# _normalize_rows
# ---------------------------------------------------------------------------


def test_normalize_rows_empty() -> None:
    assert _normalize_rows([]) == []


def test_normalize_rows_single_hit() -> None:
    hit = _make_hit(rrf_score=0.4, dense_top_score=0.9)
    rows = _normalize_rows([hit])
    assert len(rows) == 1
    assert rows[0]["hybrid_score"] == pytest.approx(0.9)


def test_normalize_rows_top_row_gets_dense_score() -> None:
    hits = [
        _make_hit(rrf_score=0.8, dense_top_score=0.75),
        _make_hit(rrf_score=0.4, dense_top_score=0.75),
    ]
    rows = _normalize_rows(hits)
    # Top row must equal dense_top_score
    assert rows[0]["hybrid_score"] == pytest.approx(0.75)
    # Second row proportional: 0.75 * (0.4/0.8) = 0.375
    assert rows[1]["hybrid_score"] == pytest.approx(0.375)


def test_normalize_rows_zero_rrf() -> None:
    hit = _make_hit(rrf_score=0.0, dense_top_score=0.5)
    rows = _normalize_rows([hit])
    assert rows[0]["hybrid_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# assess_retrieval_confidence
# ---------------------------------------------------------------------------


def test_confidence_gate_passes_above_threshold() -> None:
    settings = get_settings()
    rows: list[dict[str, object]] = [{"hybrid_score": 0.8}]
    assess_retrieval_confidence(rows, settings=settings, query_preview="test")


def test_confidence_gate_rejects_below_threshold() -> None:
    settings = get_settings()
    rows: list[dict[str, object]] = [{"hybrid_score": 0.1}]
    with pytest.raises(OutOfDomainQueryError):
        assess_retrieval_confidence(rows, settings=settings, query_preview="test")


def test_confidence_gate_disabled_skips_check() -> None:
    from core.config import Settings

    settings = Settings(retrieval_confidence_gate_enabled=False)  # type: ignore[call-arg]
    rows: list[dict[str, object]] = [{"hybrid_score": 0.0}]
    # Should not raise even with a zero score when gate is disabled
    assess_retrieval_confidence(rows, settings=settings, query_preview="test")


def test_confidence_gate_raises_on_missing_score() -> None:
    settings = get_settings()
    with pytest.raises(RetrievalError, match="hybrid_score"):
        assess_retrieval_confidence(
            [{"no_score": True}],  # type: ignore[list-item]
            settings=settings,
            query_preview="x",
        )


# ---------------------------------------------------------------------------
# HybridRetriever.retrieve (with mocks)
# ---------------------------------------------------------------------------


def _make_retriever(
    *,
    qdrant_results: list[HybridSearchResult],
    parent_records: dict[UUID, dict[str, Any]] | None = None,
    second_call_results: list[HybridSearchResult] | None = None,
    settings: Any | None = None,
) -> HybridRetriever:
    """Build a HybridRetriever with all external I/O mocked."""
    if settings is None:
        settings = get_settings()

    # Mock embedder — returns a 1024-d zero vector
    embedder = AsyncMock()
    embedder.embed.return_value = tuple(0.0 for _ in range(1024))

    # Mock sparse encoder
    sparse_encoder = AsyncMock()
    sparse_encoder.embed_query.return_value = MagicMock()

    # Mock vector store
    vector_store = AsyncMock()
    if second_call_results is not None:
        vector_store.hybrid_search.side_effect = [qdrant_results, second_call_results]
    else:
        vector_store.hybrid_search.return_value = qdrant_results

    # Mock repository
    repository = AsyncMock()
    if parent_records is None:
        parent_records = {
            _PARENT_ID: {
                "id": _PARENT_ID,
                "doc_id": _DOC_ID,
                "text_content": "Parent text content.",
                "metadata": {"source_url": "https://www.irs.gov/doc.pdf"},
            }
        }
    repository.fetch_parents.return_value = parent_records

    return HybridRetriever(
        repository=repository,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        sparse_encoder=sparse_encoder,  # type: ignore[arg-type]
        settings=settings,
    )


@pytest.mark.asyncio
async def test_retrieve_returns_context() -> None:
    hit = _make_hit(rrf_score=1.0, dense_top_score=0.9)
    retriever = _make_retriever(qdrant_results=[hit])
    ctx = await retriever.retrieve("What is the standard deduction?")
    assert len(ctx.parent_nodes) == 1
    assert ctx.parent_nodes[0].text_content == "Parent text content."
    assert len(ctx.matched_child_ids) == 1
    assert len(ctx.source_urls) == 1


@pytest.mark.asyncio
async def test_retrieve_raises_on_empty_query() -> None:
    retriever = _make_retriever(qdrant_results=[])
    with pytest.raises(RetrievalError, match="Empty"):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_retrieve_raises_no_results() -> None:
    retriever = _make_retriever(qdrant_results=[])
    with pytest.raises(RetrievalError, match="No matching"):
        await retriever.retrieve("What is the standard deduction?")


@pytest.mark.asyncio
async def test_retrieve_filter_relaxation_on_zero_results(_patch_router: AsyncMock) -> None:
    """When filters yield zero hits, a second unfiltered call is made."""
    # Override router to return a year filter so relaxation triggers.
    _patch_router.return_value = _FILTERED_ROUTE

    fallback_hit = _make_hit(rrf_score=1.0, dense_top_score=0.85)
    retriever = _make_retriever(
        qdrant_results=[],
        second_call_results=[fallback_hit],
    )
    ctx = await retriever.retrieve("Form 1040 2024 standard deduction")
    assert len(ctx.parent_nodes) == 1
    # Verify two calls were made to the vector store
    assert retriever._vector_store.hybrid_search.call_count == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_retrieve_deduplicates_parents() -> None:
    """Multiple children from the same parent collapse to one parent node."""
    from core.config import Settings

    # Use explicit settings with top_k_parents > 1 so both children from the
    # same parent are collected before the parent cap is reached.
    settings = Settings(  # type: ignore[call-arg]
        retrieval_top_k_parents=3,
        retrieval_confidence_gate_enabled=False,
    )

    child2_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    hits = [
        _make_hit(rrf_score=1.0, dense_top_score=0.8, child_id=str(_CHILD_ID)),
        _make_hit(rrf_score=0.8, dense_top_score=0.8, child_id=str(child2_id)),
    ]
    retriever = _make_retriever(qdrant_results=hits, settings=settings)
    ctx = await retriever.retrieve("deductions")
    assert len(ctx.parent_nodes) == 1
    assert len(ctx.matched_child_ids) == 2


@pytest.mark.asyncio
async def test_retrieve_rejects_low_confidence() -> None:
    """Gate raises OutOfDomainQueryError when dense cosine is too low."""
    hit = _make_hit(rrf_score=1.0, dense_top_score=0.05)
    retriever = _make_retriever(qdrant_results=[hit])
    with pytest.raises(OutOfDomainQueryError):
        await retriever.retrieve("What is today's weather?")


