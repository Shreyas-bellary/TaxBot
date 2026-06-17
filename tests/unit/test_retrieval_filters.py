"""Tests for the query-filter extractor used by the hybrid retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.retrieval import extract_filters

# ---------------------------------------------------------------------------
# Basic extraction: tax_year, form_number, doc_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, year, form, doc_type",
    [
        # Year extraction
        ("What is the 2024 standard deduction for MFJ?", 2024, None, None),
        # Form reference → doc_type must NOT be set from "Form" token alone
        ("How do I file Form 1040 in 2023?", 2023, "Form 1040", None),
        # "Publication" *is* an explicit doc-type signal
        ("Where do I report QBI on Publication 535?", None, "Publication 535", "publication"),
        # "Schedule" → form_number extracted, doc_type must NOT be forced to "form"
        ("What does Schedule SE cover?", None, "Schedule SE", None),
        # Explicit "instructions" keyword → doc_type = instruction
        ("How long can the IRS keep records per the instructions?", None, None, "instruction"),
        # "instruction" (singular) → doc_type = instruction
        ("See the instruction for completing Schedule B.", None, "Schedule B", "instruction"),
        # "pub " shorthand — Pub is also a form-number prefix, so form_number is set
        ("Refer to pub 505 for safe-harbor rules.", None, "Pub 505", "publication"),
        # "notice" keyword — use a query without a year-like digit sequence
        ("What does the IRS notice regarding EV credits say?", None, None, "notice"),
    ],
)
def test_extract_filters_basic(
    query: str, year: int | None, form: str | None, doc_type: str | None
) -> None:
    filters = extract_filters(query)
    assert filters.tax_year == year
    assert filters.form_number == form
    assert filters.doc_type == doc_type


# ---------------------------------------------------------------------------
# case_03 regression: "Form 2555 instruct" must NOT set doc_type='form'
# ---------------------------------------------------------------------------


def test_case_03_form_2555_does_not_set_form_doc_type() -> None:
    """Bare 'Form NNNN' in a query must never trigger doc_type='form'.

    Previously the naive 'form' hint would match the substring and restrict
    retrieval to only the blank-form PDF, missing the Instructions PDF where
    the 330-day rule is defined.
    """
    query = (
        "How does Form 2555 instruct a US citizen working abroad to determine "
        "whether they meet the physical presence test for the foreign earned "
        "income exclusion?"
    )
    filters = extract_filters(query)
    assert filters.form_number == "Form 2555"
    assert filters.doc_type != "form", (
        "doc_type must not be 'form' — 'Form' is a proper-noun reference, not a doc-type signal"
    )


# ---------------------------------------------------------------------------
# form_number_variants: form-family MatchAny set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_variants",
    [
        # Form reference → blank form + IRS instruction product number
        (
            "How does Form 2555 instruct you to determine the physical presence test?",
            ("Form 2555", "Instruction 2555"),
        ),
        # Schedule reference → schedule + instruction product number
        (
            "How do I complete Schedule SE?",
            ("Schedule SE", "Instruction SE"),
        ),
        # Form 1040 → includes instruction variant
        (
            "How do I file Form 1040?",
            ("Form 1040", "Instruction 1040"),
        ),
        # Hyphenated form suffix preserved in instruction product number
        (
            "How do I complete Form 706-A?",
            ("Form 706-A", "Instruction 706-A"),
        ),
        # Publication → only the publication itself (no instructions variant)
        (
            "Where is QBI explained in Publication 535?",
            ("Publication 535",),
        ),
        # No form reference → no variants
        (
            "What is the standard deduction for MFJ in 2024?",
            (),
        ),
    ],
)
def test_form_number_variants(query: str, expected_variants: tuple[str, ...]) -> None:
    filters = extract_filters(query)
    assert filters.form_number_variants == expected_variants


# ---------------------------------------------------------------------------
# procedural_intent detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expect_procedural",
    [
        # "how does ... instruct" → procedural
        (
            "How does Form 2555 instruct a taxpayer to determine the 330-day test?",
            True,
        ),
        # "how to" → procedural
        ("How to report foreign income on Form 2555?", True),
        # "how do I" → procedural
        ("How do I complete Schedule SE?", True),
        # "determine" → procedural
        ("What thresholds determine the AMT exemption phase-out?", True),
        # "complete" → procedural
        ("Where do I complete the carryover section?", True),
        # factual lookup → not procedural
        ("What is the 2024 standard deduction for married filing jointly?", False),
        # publication reference → not procedural
        ("What does Publication 535 say about QBI?", False),
    ],
)
def test_procedural_intent(query: str, expect_procedural: bool) -> None:
    filters = extract_filters(query)
    assert filters.procedural_intent is expect_procedural


# ---------------------------------------------------------------------------
# QueryFilters is fully populated (no missing fields)
# ---------------------------------------------------------------------------


def test_query_filters_all_fields_present() -> None:
    """Ensure QueryFilters exposes all expected fields."""
    f = extract_filters("How does Form 2555 instruct you to determine the 330-day test?")
    assert hasattr(f, "tax_year")
    assert hasattr(f, "form_number")
    assert hasattr(f, "doc_type")
    assert hasattr(f, "form_number_variants")
    assert hasattr(f, "procedural_intent")


# ---------------------------------------------------------------------------
# _build_metadata_filter: MatchAny for form-family variants
# ---------------------------------------------------------------------------


def test_build_filter_uses_match_any_for_variants() -> None:
    from qdrant_client import models

    from core.vector_store import _build_metadata_filter

    result = _build_metadata_filter(
        tax_year=None,
        form_number="Form 2555",
        doc_type=None,
        form_number_variants=("Form 2555", "Instruction 2555"),
    )
    assert result is not None
    conds = result.must  # type: ignore[union-attr]
    assert len(conds) == 1
    form_cond = conds[0]
    assert form_cond.key == "form_number"
    # Must be MatchAny, not MatchValue
    assert isinstance(form_cond.match, models.MatchAny), (
        "form_number_variants must produce MatchAny, not MatchValue"
    )
    assert set(form_cond.match.any) == {"Form 2555", "Instruction 2555"}


def test_build_filter_match_any_covers_both_form_and_instructions() -> None:
    """MatchAny filter for Form 2555 variants must include the instructions label."""
    from core.vector_store import _build_metadata_filter

    result = _build_metadata_filter(
        tax_year=None,
        form_number="Form 2555",
        doc_type=None,
        form_number_variants=("Form 2555", "Instruction 2555"),
    )
    assert result is not None
    form_cond = result.must[0]  # type: ignore[index]
    assert "Instruction 2555" in form_cond.match.any  # type: ignore[union-attr]


def test_build_filter_variants_take_precedence_over_bare_form_number() -> None:
    """When variants are provided, the bare form_number value is not used."""
    from qdrant_client import models

    from core.vector_store import _build_metadata_filter

    # If both form_number and variants are given, variants win
    result = _build_metadata_filter(
        tax_year=None,
        form_number="Form 2555",
        doc_type=None,
        form_number_variants=("Form 2555", "Instruction 2555"),
    )
    assert result is not None
    form_cond = result.must[0]  # type: ignore[index]
    # Must be MatchAny (not MatchValue with single string)
    assert isinstance(form_cond.match, models.MatchAny)


def test_build_filter_bare_form_number_fallback() -> None:
    """Without variants, bare form_number still produces a MatchValue filter."""
    from qdrant_client import models

    from core.vector_store import _build_metadata_filter

    result = _build_metadata_filter(
        tax_year=None,
        form_number="Publication 535",
        doc_type=None,
    )
    assert result is not None
    form_cond = result.must[0]  # type: ignore[index]
    assert isinstance(form_cond.match, models.MatchValue)
    assert form_cond.match.value == "Publication 535"


def test_build_filter_doc_type_combined_with_variants() -> None:
    """doc_type filter is AND-ed with form_number_variants when both are set."""
    from core.vector_store import _build_metadata_filter

    result = _build_metadata_filter(
        tax_year=None,
        form_number=None,
        doc_type="instruction",
        form_number_variants=("Instruction 2555",),
    )
    assert result is not None
    keys = {c.key for c in result.must}  # type: ignore[union-attr]
    assert "form_number" in keys
    assert "doc_type" in keys


# ---------------------------------------------------------------------------
# Tiered relaxation: stage-1 procedural search hits instruction variant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_procedural_tries_instruction_variant_first() -> None:
    """When procedural_intent=True, retrieve() tries doc_type=instruction first."""
    from core.retrieval import HybridRetriever

    mock_repo = AsyncMock()
    mock_embedder = AsyncMock()
    mock_embedder.embed.return_value = tuple([0.1] * 1024)
    mock_vs = AsyncMock()
    mock_sparse = AsyncMock()

    from qdrant_client.models import SparseVector

    mock_sparse.embed_query.return_value = SparseVector(indices=[1], values=[1.0])

    _CHILD_UUID = "00000000-0000-0000-0000-000000000002"
    _PARENT_UUID = "00000000-0000-0000-0000-000000000001"

    # Simulate: instruction-only search returns 1 result; standard search returns 2
    instruction_hit = MagicMock()
    instruction_hit.child_id = _CHILD_UUID
    instruction_hit.parent_id = _PARENT_UUID
    instruction_hit.doc_id = "00000000-0000-0000-0000-000000000003"
    instruction_hit.rrf_score = 0.9
    instruction_hit.dense_top_score = 0.85

    # First call (instruction-only) returns a hit; second call should NOT be reached
    mock_vs.hybrid_search.return_value = [instruction_hit]

    import uuid as _uuid

    mock_repo.fetch_parents.return_value = {
        _uuid.UUID(_PARENT_UUID): {
            "doc_id": "00000000-0000-0000-0000-000000000003",
            "text_content": "You must be physically present...",
            "metadata": {"source_url": "https://irs.gov/f2555i.pdf"},
        }
    }

    from core.config import Settings

    settings = Settings(
        postgres_dsn="postgresql://u:p@localhost/db",
        unstructured_api_key="x",
        huggingface_api_token="x",
        gemini_api_key="x",
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key="x",
        retrieval_confidence_gate_enabled=False,
    )

    retriever = HybridRetriever(
        repository=mock_repo,
        embedder=mock_embedder,
        vector_store=mock_vs,
        sparse_encoder=mock_sparse,
        settings=settings,
    )

    query = (
        "How does Form 2555 instruct a US citizen working abroad to determine "
        "whether they meet the physical presence test?"
    )
    ctx = await retriever.retrieve(query)

    # Confirm hybrid_search was called with instruction variant + doc_type
    first_call_kwargs = mock_vs.hybrid_search.call_args_list[0].kwargs
    assert first_call_kwargs.get("doc_type") == "instruction"
    assert "Instruction 2555" in first_call_kwargs.get(
        "form_number_variants", ()
    )
    assert ctx is not None
