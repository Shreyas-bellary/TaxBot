"""Tests for deterministic text splitting helpers."""

from __future__ import annotations

from ingestion.text_splitter import (
    group_narratives_into_parents,
    split_into_child_sentences,
)
from ingestion.unstructured_parser import NarrativeBlock


def _block(text: str) -> NarrativeBlock:
    return NarrativeBlock(
        element_id="x",
        element_type="NarrativeText",
        text=text,
        page_number=1,
        section=None,
    )


def test_group_narratives_into_parents_packs_until_target() -> None:
    short = "Short sentence about the tax year."
    blocks = [_block(short) for _ in range(80)]
    parents = group_narratives_into_parents(
        blocks,
        target_chars=500,
        max_chars=900,
    )
    assert parents
    assert all(len(p) >= 1 for p in parents)
    assert max(len(p) for p in parents) <= 900 + len(short)


def test_split_into_child_sentences_handles_empty_and_short() -> None:
    assert split_into_child_sentences("") == []
    assert split_into_child_sentences("Single very short.") == ["Single very short."]


def test_split_into_child_sentences_chunks_long_text() -> None:
    text = " ".join(
        [
            "The Earned Income Tax Credit is a refundable credit for low and moderate income workers.",
            "It requires earned income and adjusted gross income to be below certain thresholds.",
            "These thresholds vary by filing status and the number of qualifying children.",
            "Investment income above the annual limit disqualifies the taxpayer.",
            "The maximum credit is indexed for inflation each year.",
        ]
    )
    chunks = split_into_child_sentences(text, target_chars=120, max_chars=200)
    assert len(chunks) >= 2
    assert all(len(c) <= 200 + 30 for c in chunks)
    assert "".join(chunks).replace(" ", "").startswith("TheEarnedIncomeTaxCredit")
