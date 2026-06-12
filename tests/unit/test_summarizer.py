"""Tests for the table summariser primary/fallback logic and post-validator."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from core.config import Settings
from core.errors import SummarizationError
from ingestion.summarizer import (
    TableSummarizer,
    TableSummaryInput,
    _enforce_three_sentences,
    _resolve_table_summarizer_providers,
)


def test_enforce_three_sentences_trims_to_three() -> None:
    text = (
        "First sentence describing the table. Second sentence about axes. "
        "Third sentence about takeaways. Fourth sentence we should drop."
    )
    out = _enforce_three_sentences(text)
    assert out.count(".") == 3


def test_enforce_three_sentences_rejects_empty() -> None:
    with pytest.raises(SummarizationError):
        _enforce_three_sentences("   ")


def test_enforce_three_sentences_rejects_too_short_sentence() -> None:
    with pytest.raises(SummarizationError):
        _enforce_three_sentences("Hi. Yo. Hey.")


@pytest.mark.asyncio
async def test_summarizer_falls_back_when_primary_fails() -> None:
    async def primary(_: TableSummaryInput) -> str:
        raise RuntimeError("primary down")

    async def fallback(_: TableSummaryInput) -> str:
        return (
            "This is a tax bracket table for 2024. "
            "Columns enumerate filing statuses and tax rates. "
            "The top bracket of 37 percent kicks in above $609,350 for single filers."
        )

    summarizer = TableSummarizer(primary=primary, fallback=fallback)
    payload = TableSummaryInput(
        doc_number="Form 1040",
        doc_title="U.S. Individual Income Tax Return",
        tax_year=2024,
        table_markdown="| col | rate |\n| --- | --- |\n| top | 37% |",
    )
    text = await summarizer.summarize(payload)
    assert "2024" in text


@pytest.mark.asyncio
async def test_summarizer_raises_when_all_paths_fail() -> None:
    async def primary(_: TableSummaryInput) -> str:
        raise RuntimeError("primary boom")

    async def fallback(_: TableSummaryInput) -> str:
        raise RuntimeError("fallback boom")

    summarizer = TableSummarizer(primary=primary, fallback=fallback)
    with pytest.raises(SummarizationError):
        await summarizer.summarize(
            TableSummaryInput(
                doc_number="Form 1040",
                doc_title="U.S. Individual Income Tax Return",
                tax_year=2024,
                table_markdown="| a | b |\n| - | - |\n| 1 | 2 |",
            )
        )


def test_resolve_table_summarizer_providers_gemini_primary() -> None:
    settings = Settings(
        postgres_dsn="postgresql://postgres:postgres@localhost:5432/postgres",
        unstructured_api_key=SecretStr("key"),
        huggingface_api_token=SecretStr("token"),
        gemini_api_key=SecretStr("gemini"),
        openrouter_api_key=SecretStr("openrouter"),
        table_summary_provider="gemini",
    )
    primary, fallback = _resolve_table_summarizer_providers(settings)
    assert primary is not None
    assert fallback is not None


def test_resolve_table_summarizer_providers_openrouter_primary() -> None:
    settings = Settings(
        postgres_dsn="postgresql://postgres:postgres@localhost:5432/postgres",
        unstructured_api_key=SecretStr("key"),
        huggingface_api_token=SecretStr("token"),
        gemini_api_key=SecretStr("gemini"),
        openrouter_api_key=SecretStr("openrouter"),
        table_summary_provider="openrouter",
    )
    primary, fallback = _resolve_table_summarizer_providers(settings)
    assert primary is not None
    assert fallback is not None


def test_resolve_table_summarizer_providers_openrouter_requires_key() -> None:
    settings = Settings(
        postgres_dsn="postgresql://postgres:postgres@localhost:5432/postgres",
        unstructured_api_key=SecretStr("key"),
        huggingface_api_token=SecretStr("token"),
        gemini_api_key=SecretStr("gemini"),
        openrouter_api_key=None,
        table_summary_provider="openrouter",
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        _resolve_table_summarizer_providers(settings)
