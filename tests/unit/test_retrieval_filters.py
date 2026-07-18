"""Tests for the LLM query router and Qdrant filter construction.

The query router replaces all regex-based filter extraction.  These tests
cover:

- ``_build_metadata_filter`` in :mod:`core.vector_store` (no network calls)
- ``_parse_router_response`` in :mod:`core.query_router` (no network calls)
- ``route_query`` with a mocked LLM (no real API calls)
- ``HybridRetriever.retrieve`` with a mocked router + vector store
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from core.errors import OutOfDomainQueryError, RouterError
from core.query_router import (
    RouteFilters,
    _parse_router_response,
)
from core.vector_store import _build_metadata_filter

# ---------------------------------------------------------------------------
# _build_metadata_filter — pure unit tests, no network
# ---------------------------------------------------------------------------


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


def test_filter_single_form_number_uses_match_value() -> None:
    result = _build_metadata_filter(
        tax_year=None, doc_type=None, form_numbers=["Form 2555"]
    )
    assert result is not None
    cond = result.must[0]  # type: ignore[index]
    assert cond.key == "form_number"
    assert isinstance(cond.match, models.MatchValue)
    assert cond.match.value == "Form 2555"


def test_filter_multiple_form_numbers_uses_match_any() -> None:
    result = _build_metadata_filter(
        tax_year=None,
        doc_type=None,
        form_numbers=["Form 2555", "Instruction 2555"],
    )
    assert result is not None
    cond = result.must[0]  # type: ignore[index]
    assert cond.key == "form_number"
    assert isinstance(cond.match, models.MatchAny)
    assert set(cond.match.any) == {"Form 2555", "Instruction 2555"}


def test_filter_doc_type_only() -> None:
    result = _build_metadata_filter(
        tax_year=None, doc_type="instruction", form_numbers=None
    )
    assert result is not None
    cond = result.must[0]  # type: ignore[index]
    assert cond.key == "doc_type"
    assert isinstance(cond.match, models.MatchValue)
    assert cond.match.value == "instruction"


def test_filter_all_fields_combined() -> None:
    result = _build_metadata_filter(
        tax_year=2023,
        doc_type="publication",
        form_numbers=["Publication 17"],
    )
    assert result is not None
    keys = {c.key for c in result.must}  # type: ignore[union-attr]
    assert keys == {"tax_year", "form_number", "doc_type"}


def test_filter_empty_form_numbers_list_is_no_filter() -> None:
    result = _build_metadata_filter(tax_year=None, doc_type=None, form_numbers=[])
    assert result is None


# ---------------------------------------------------------------------------
# _parse_router_response — pure unit tests, no network
# ---------------------------------------------------------------------------


def test_parse_in_domain_with_filters() -> None:
    raw = json.dumps(
        {
            "in_domain": True,
            "filters": {
                "tax_year": 2024,
                "doc_type": "instruction",
                "form_numbers": ["Form 1040", "Instruction 1040"],
            },
        }
    )
    result = _parse_router_response(raw)
    assert result.in_domain is True
    assert result.filters is not None
    assert result.filters.tax_year == 2024
    assert result.filters.doc_type == "instruction"
    assert result.filters.form_numbers == ["Form 1040", "Instruction 1040"]


def test_parse_in_domain_null_filters() -> None:
    raw = json.dumps({"in_domain": True, "filters": None})
    result = _parse_router_response(raw)
    assert result.in_domain is True
    assert result.filters is None


def test_parse_out_of_domain() -> None:
    raw = json.dumps({"in_domain": False, "filters": None})
    result = _parse_router_response(raw)
    assert result.in_domain is False


def test_parse_strips_markdown_fences() -> None:
    raw = "```json\n" + json.dumps({"in_domain": True, "filters": {}}) + "\n```"
    result = _parse_router_response(raw)
    assert result.in_domain is True


def test_parse_raises_router_error_on_bad_json() -> None:
    with pytest.raises(RouterError, match="non-JSON"):
        _parse_router_response("not json at all")


def test_parse_raises_router_error_on_missing_in_domain() -> None:
    with pytest.raises(RouterError, match="validation"):
        _parse_router_response(json.dumps({"filters": None}))


# ---------------------------------------------------------------------------
# route_query — mocked LLM, no real API calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_query_in_domain_returns_filters() -> None:
    from core.query_router import route_query
    from core.security import SanitizedQuery

    san = SanitizedQuery(
        cleaned_text="How does Form 2555 determine the physical presence test?",
        fenced_prompt_section="[START]\nHow does Form 2555 ...\n[END]",
        start_tag="START",
        end_tag="END",
    )

    raw_response = json.dumps(
        {
            "in_domain": True,
            "filters": {
                "tax_year": None,
                "doc_type": "instruction",
                "form_numbers": ["Form 2555", "Instruction 2555"],
            },
        }
    )

    with patch("core.query_router._call_gemini", new=AsyncMock(return_value=raw_response)):
        from core.config import Settings

        settings = Settings(
            postgres_dsn="postgresql://u:p@localhost/db",
            unstructured_api_key="x",
            huggingface_api_token="x",
            gemini_api_key="x",
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="x",
            router_llm_provider="gemini",
            router_llm_model="gemini-2.0-flash",
        )
        result = await route_query(san, settings=settings)

    assert result.filters.doc_type == "instruction"
    assert "Form 2555" in (result.filters.form_numbers or [])
    assert "Instruction 2555" in (result.filters.form_numbers or [])


@pytest.mark.asyncio
async def test_route_query_out_of_domain_raises() -> None:
    from core.query_router import route_query
    from core.security import SanitizedQuery

    san = SanitizedQuery(
        cleaned_text="What is the weather in New York?",
        fenced_prompt_section="[START]\nWhat is the weather in New York?\n[END]",
        start_tag="START",
        end_tag="END",
    )

    raw_response = json.dumps({"in_domain": False, "filters": None})

    with patch("core.query_router._call_gemini", new=AsyncMock(return_value=raw_response)):
        from core.config import Settings

        settings = Settings(
            postgres_dsn="postgresql://u:p@localhost/db",
            unstructured_api_key="x",
            huggingface_api_token="x",
            gemini_api_key="x",
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="x",
            router_llm_provider="gemini",
            router_llm_model="gemini-2.0-flash",
        )
        with pytest.raises(OutOfDomainQueryError):
            await route_query(san, settings=settings)


@pytest.mark.asyncio
async def test_route_query_router_error_propagates() -> None:
    from core.query_router import route_query
    from core.security import SanitizedQuery

    san = SanitizedQuery(
        cleaned_text="What is the 2024 standard deduction?",
        fenced_prompt_section="[START]\nWhat is the 2024 standard deduction?\n[END]",
        start_tag="START",
        end_tag="END",
    )

    with patch(
        "core.query_router._call_gemini",
        new=AsyncMock(return_value="not valid json {{{{"),
    ):
        from core.config import Settings

        settings = Settings(
            postgres_dsn="postgresql://u:p@localhost/db",
            unstructured_api_key="x",
            huggingface_api_token="x",
            gemini_api_key="x",
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="x",
            router_llm_provider="gemini",
            router_llm_model="gemini-2.0-flash",
        )
        with pytest.raises(RouterError):
            await route_query(san, settings=settings)


@pytest.mark.asyncio
async def test_route_query_includes_history_in_user_message() -> None:
    from core.conversation import ChatTurn
    from core.query_router import route_query
    from core.security import SanitizedQuery

    san = SanitizedQuery(
        cleaned_text="what about for 2025?",
        fenced_prompt_section="[START]\nwhat about for 2025?\n[END]",
        start_tag="START",
        end_tag="END",
    )
    history = (
        ChatTurn(
            role="user",
            content="What is the standard deduction for tax year 2024?",
        ),
        ChatTurn(
            role="assistant",
            content="The standard deduction for tax year 2024 is $29,200.",
        ),
    )
    raw_response = json.dumps(
        {
            "in_domain": True,
            "filters": {"tax_year": 2025, "doc_type": None, "form_numbers": None},
            "retrieval_query": "What is the standard deduction for tax year 2025?",
        }
    )
    call_mock = AsyncMock(return_value=raw_response)

    with patch("core.query_router._call_gemini", new=call_mock):
        from core.config import Settings

        settings = Settings(
            postgres_dsn="postgresql://u:p@localhost/db",
            unstructured_api_key="x",
            huggingface_api_token="x",
            gemini_api_key="x",
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="x",
            router_llm_provider="gemini",
            router_llm_model="gemini-2.0-flash",
        )
        result = await route_query(san, history=history, settings=settings)

    assert result.filters.tax_year == 2025
    assert result.filters.form_numbers is None
    assert result.retrieval_query == (
        "What is the standard deduction for tax year 2025?"
    )
    user_message = call_mock.await_args.kwargs["user_message"]
    assert "standard deduction" in user_message.lower()
    assert "what about for 2025?" in user_message


@pytest.mark.asyncio
async def test_retrieve_uses_router_retrieval_query() -> None:
    """Vague follow-ups are embedded using the router's rewritten query."""
    import uuid

    from core.retrieval import HybridRetriever

    _CHILD_UUID = "00000000-0000-0000-0000-000000000002"
    _PARENT_UUID = "00000000-0000-0000-0000-000000000001"

    mock_hit = MagicMock()
    mock_hit.child_id = _CHILD_UUID
    mock_hit.parent_id = _PARENT_UUID
    mock_hit.doc_id = "00000000-0000-0000-0000-000000000003"
    mock_hit.rrf_score = 0.9
    mock_hit.dense_top_score = 0.85

    mock_vs = AsyncMock()
    mock_vs.hybrid_search.return_value = [mock_hit]

    mock_repo = AsyncMock()
    mock_repo.fetch_parents.return_value = {
        uuid.UUID(_PARENT_UUID): {
            "doc_id": "00000000-0000-0000-0000-000000000003",
            "text_content": "2025 standard deduction...",
            "metadata": {"source_url": "https://irs.gov/pub/irs-pdf/p17.pdf"},
        }
    }

    mock_embedder = AsyncMock()
    mock_embedder.embed.return_value = tuple([0.1] * 1024)
    mock_sparse = AsyncMock()
    from qdrant_client.models import SparseVector
    mock_sparse.embed_query.return_value = SparseVector(indices=[1], values=[1.0])

    from core.config import Settings
    from core.query_router import QueryRouteResult

    settings = Settings(
        postgres_dsn="postgresql://u:p@localhost/db",
        unstructured_api_key="x",
        huggingface_api_token="x",
        gemini_api_key="x",
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key="x",
        retrieval_confidence_gate_enabled=False,
        router_llm_provider="gemini",
        router_llm_model="gemini-2.0-flash",
    )

    rewritten = "What is the standard deduction for tax year 2025?"
    route_result = QueryRouteResult(
        filters=RouteFilters(tax_year=2025, form_numbers=None),
        retrieval_query=rewritten,
    )

    with patch("core.retrieval.route_query", new=AsyncMock(return_value=route_result)):
        retriever = HybridRetriever(
            repository=mock_repo,
            embedder=mock_embedder,
            vector_store=mock_vs,
            sparse_encoder=mock_sparse,
            settings=settings,
        )
        ctx = await retriever.retrieve("what about 2025?")

    mock_embedder.embed.assert_awaited_once_with(rewritten)
    mock_sparse.embed_query.assert_awaited_once_with(rewritten)
    assert ctx.query == rewritten
    call_kwargs = mock_vs.hybrid_search.call_args_list[0].kwargs
    assert call_kwargs.get("tax_year") == 2025
    assert call_kwargs.get("form_numbers") is None


# ---------------------------------------------------------------------------
# HybridRetriever.retrieve — mocked router + vector store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_applies_router_filters() -> None:
    """retrieve() calls hybrid_search with filters from the router."""
    import uuid

    from core.retrieval import HybridRetriever

    _CHILD_UUID = "00000000-0000-0000-0000-000000000002"
    _PARENT_UUID = "00000000-0000-0000-0000-000000000001"

    mock_hit = MagicMock()
    mock_hit.child_id = _CHILD_UUID
    mock_hit.parent_id = _PARENT_UUID
    mock_hit.doc_id = "00000000-0000-0000-0000-000000000003"
    mock_hit.rrf_score = 0.9
    mock_hit.dense_top_score = 0.85

    mock_vs = AsyncMock()
    mock_vs.hybrid_search.return_value = [mock_hit]

    mock_repo = AsyncMock()
    mock_repo.fetch_parents.return_value = {
        uuid.UUID(_PARENT_UUID): {
            "doc_id": "00000000-0000-0000-0000-000000000003",
            "text_content": "You must be physically present...",
            "metadata": {"source_url": "https://irs.gov/pub/irs-pdf/i2555.pdf"},
        }
    }

    mock_embedder = AsyncMock()
    mock_embedder.embed.return_value = tuple([0.1] * 1024)
    mock_sparse = AsyncMock()
    from qdrant_client.models import SparseVector
    mock_sparse.embed_query.return_value = SparseVector(indices=[1], values=[1.0])

    from core.config import Settings
    from core.query_router import QueryRouteResult

    settings = Settings(
        postgres_dsn="postgresql://u:p@localhost/db",
        unstructured_api_key="x",
        huggingface_api_token="x",
        gemini_api_key="x",
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key="x",
        retrieval_confidence_gate_enabled=False,
        router_llm_provider="gemini",
        router_llm_model="gemini-2.0-flash",
    )

    route_result = QueryRouteResult(
        filters=RouteFilters(
            tax_year=None,
            doc_type="instruction",
            form_numbers=["Form 2555", "Instruction 2555"],
        )
    )

    with patch("core.retrieval.route_query", new=AsyncMock(return_value=route_result)):
        retriever = HybridRetriever(
            repository=mock_repo,
            embedder=mock_embedder,
            vector_store=mock_vs,
            sparse_encoder=mock_sparse,
            settings=settings,
        )
        ctx = await retriever.retrieve(
            "How does Form 2555 determine the physical presence test?"
        )

    call_kwargs = mock_vs.hybrid_search.call_args_list[0].kwargs
    assert call_kwargs.get("doc_type") == "instruction"
    assert call_kwargs.get("form_numbers") == ["Form 2555", "Instruction 2555"]
    assert ctx is not None


@pytest.mark.asyncio
async def test_retrieve_relaxes_filters_on_zero_hits() -> None:
    """retrieve() retries without filters when filtered search returns nothing."""
    import uuid

    from core.retrieval import HybridRetriever

    _PARENT_UUID = "00000000-0000-0000-0000-000000000001"
    _CHILD_UUID = "00000000-0000-0000-0000-000000000002"

    mock_hit = MagicMock()
    mock_hit.child_id = _CHILD_UUID
    mock_hit.parent_id = _PARENT_UUID
    mock_hit.doc_id = "00000000-0000-0000-0000-000000000003"
    mock_hit.rrf_score = 0.9
    mock_hit.dense_top_score = 0.85

    mock_vs = AsyncMock()
    # First call (filtered) → empty; second (unfiltered) → hit
    mock_vs.hybrid_search.side_effect = [[], [mock_hit]]

    mock_repo = AsyncMock()
    mock_repo.fetch_parents.return_value = {
        uuid.UUID(_PARENT_UUID): {
            "doc_id": "00000000-0000-0000-0000-000000000003",
            "text_content": "Standard deduction for MFJ is $29,200.",
            "metadata": {"source_url": "https://irs.gov/pub/irs-pdf/p17.pdf"},
        }
    }

    mock_embedder = AsyncMock()
    mock_embedder.embed.return_value = tuple([0.1] * 1024)
    mock_sparse = AsyncMock()
    from qdrant_client.models import SparseVector
    mock_sparse.embed_query.return_value = SparseVector(indices=[1], values=[1.0])

    from core.config import Settings
    from core.query_router import QueryRouteResult

    settings = Settings(
        postgres_dsn="postgresql://u:p@localhost/db",
        unstructured_api_key="x",
        huggingface_api_token="x",
        gemini_api_key="x",
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key="x",
        retrieval_confidence_gate_enabled=False,
        router_llm_provider="gemini",
        router_llm_model="gemini-2.0-flash",
    )

    route_result = QueryRouteResult(
        filters=RouteFilters(tax_year=2024, doc_type=None, form_numbers=None)
    )

    with patch("core.retrieval.route_query", new=AsyncMock(return_value=route_result)):
        retriever = HybridRetriever(
            repository=mock_repo,
            embedder=mock_embedder,
            vector_store=mock_vs,
            sparse_encoder=mock_sparse,
            settings=settings,
        )
        ctx = await retriever.retrieve(
            "What is the 2024 standard deduction for MFJ?"
        )

    assert mock_vs.hybrid_search.call_count == 2
    # Second call must have no filters at all
    second_call = mock_vs.hybrid_search.call_args_list[1].kwargs
    assert second_call.get("tax_year") is None
    assert second_call.get("doc_type") is None
    assert second_call.get("form_numbers") is None
    assert ctx is not None


@pytest.mark.asyncio
async def test_retrieve_router_error_falls_back_to_unfiltered() -> None:
    """When the router raises RouterError, retrieve() runs unfiltered."""
    import uuid

    from core.retrieval import HybridRetriever

    _PARENT_UUID = "00000000-0000-0000-0000-000000000001"
    _CHILD_UUID = "00000000-0000-0000-0000-000000000002"

    mock_hit = MagicMock()
    mock_hit.child_id = _CHILD_UUID
    mock_hit.parent_id = _PARENT_UUID
    mock_hit.doc_id = "00000000-0000-0000-0000-000000000003"
    mock_hit.rrf_score = 0.9
    mock_hit.dense_top_score = 0.85

    mock_vs = AsyncMock()
    mock_vs.hybrid_search.return_value = [mock_hit]

    mock_repo = AsyncMock()
    mock_repo.fetch_parents.return_value = {
        uuid.UUID(_PARENT_UUID): {
            "doc_id": "00000000-0000-0000-0000-000000000003",
            "text_content": "QBI phase-out starts at $182,100.",
            "metadata": {"source_url": "https://irs.gov/pub/irs-pdf/p535.pdf"},
        }
    }

    mock_embedder = AsyncMock()
    mock_embedder.embed.return_value = tuple([0.1] * 1024)
    mock_sparse = AsyncMock()
    from qdrant_client.models import SparseVector
    mock_sparse.embed_query.return_value = SparseVector(indices=[1], values=[1.0])

    from core.config import Settings

    settings = Settings(
        postgres_dsn="postgresql://u:p@localhost/db",
        unstructured_api_key="x",
        huggingface_api_token="x",
        gemini_api_key="x",
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key="x",
        retrieval_confidence_gate_enabled=False,
        router_llm_provider="gemini",
        router_llm_model="gemini-2.0-flash",
    )

    with patch(
        "core.retrieval.route_query",
        new=AsyncMock(side_effect=RouterError("LLM timeout")),
    ):
        retriever = HybridRetriever(
            repository=mock_repo,
            embedder=mock_embedder,
            vector_store=mock_vs,
            sparse_encoder=mock_sparse,
            settings=settings,
        )
        ctx = await retriever.retrieve("What were the 2023 QBI thresholds?")

    # Should call hybrid_search once with no filters (router fallback)
    assert mock_vs.hybrid_search.call_count == 1
    call_kwargs = mock_vs.hybrid_search.call_args_list[0].kwargs
    assert call_kwargs.get("tax_year") is None
    assert call_kwargs.get("doc_type") is None
    assert call_kwargs.get("form_numbers") is None
    assert ctx is not None
