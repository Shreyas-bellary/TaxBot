"""Tests for trivial-table heuristics used during ingest."""

from __future__ import annotations

import pytest

from core.errors import SummarizationError, TrivialTableError
from ingestion.table_filters import (
    is_summarization_validation_error,
    is_trivial_table,
    should_drop_table_after_summary_failure,
)


def test_is_trivial_table_single_word_cell() -> None:
    assert is_trivial_table("| Incorrect). |")


def test_is_trivial_table_real_bracket_table() -> None:
    markdown = (
        "| Bracket | Rate |\n"
        "| --- | --- |\n"
        "| $0-$10k | 10% |\n"
        "| $10k+ | 12% |"
    )
    assert not is_trivial_table(markdown)


def test_is_summarization_validation_error() -> None:
    assert is_summarization_validation_error(
        SummarizationError("Summary sentence too short to be informative: 'x'.")
    )
    assert not is_summarization_validation_error(RuntimeError("network down"))


def test_should_drop_table_after_summary_failure() -> None:
    markdown = "| Incorrect). |"
    assert should_drop_table_after_summary_failure(
        markdown,
        SummarizationError("Summary sentence too short to be informative: 'Incorrect).'"),
    )
    assert should_drop_table_after_summary_failure(
        markdown,
        TrivialTableError("too small"),
    )


@pytest.mark.asyncio
async def test_summarizer_skips_fallback_for_trivial_table() -> None:
    from ingestion.summarizer import TableSummarizer, TableSummaryInput

    fallback_called = False

    async def primary(_: TableSummaryInput) -> str:
        raise SummarizationError("Summary sentence too short to be informative: 'Incorrect).'")

    async def fallback(_: TableSummaryInput) -> str:
        nonlocal fallback_called
        fallback_called = True
        return (
            "Fallback sentence one here. "
            "Fallback sentence two here. "
            "Fallback sentence three here."
        )

    summarizer = TableSummarizer(primary=primary, fallback=fallback)
    with pytest.raises(TrivialTableError):
        await summarizer.summarize(
            TableSummaryInput(
                doc_number="Publication 1586",
                doc_title="Example",
                tax_year=2024,
                table_markdown="| Incorrect). |",
            )
        )
    assert fallback_called is False
