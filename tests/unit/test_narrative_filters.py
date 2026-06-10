"""Tests for Layer 1 deterministic narrative content hygiene."""

from __future__ import annotations

from core.config import Settings, get_settings
from ingestion.narrative_filters import (
    filter_irs_narratives,
    trim_leading_irs_junk,
    trim_trailing_index,
)
from ingestion.unstructured_parser import NarrativeBlock


def _block(
    text: str,
    *,
    element_type: str = "NarrativeText",
    page_number: int | None = None,
    element_id: str | None = None,
) -> NarrativeBlock:
    return NarrativeBlock(
        element_id=element_id or text[:8],
        element_type=element_type,
        text=text,
        page_number=page_number,
        section=None,
    )


def _settings(*, enabled: bool) -> Settings:
    return get_settings().model_copy(
        update={"narrative_content_filter_enabled": enabled}
    )


def test_leading_junk_removed_title_kept() -> None:
    narratives = [
        _block("Page 1 of 12"),
        _block(
            "The type and rule above prints on all proofs including departmental "
            "reproduction proofs. MUST be removed before printing."
        ),
        _block("AH XSL/XML Fileid: m-1040-c/202601/a/xml/cycle05/source"),
        _block("Userid: CPM"),
        _block("Schema:"),
        _block("instrx"),
        _block("Leadpct: 100% Pt. size: 10"),
        _block("Ok to Print"),
        _block("11:41 - 2-Feb-2026"),
        _block("Instructions for Form 1040-C", element_type="Title"),
        _block("Use this form if you are a departing alien."),
    ]

    filtered = filter_irs_narratives(narratives)

    assert filtered[0].text == "Instructions for Form 1040-C"
    assert filtered[-1].text == "Use this form if you are a departing alien."
    assert len(filtered) == 2


def test_trailing_index_truncated() -> None:
    body = [_block(f"Body paragraph {i}.", page_number=1) for i in range(18)]
    narratives = [
        *body,
        _block("Index", element_type="Title", page_number=12),
        _block("Adjusted gross income, 5", page_number=12),
        _block("Credits, 9", page_number=12),
    ]

    kept, dropped = trim_trailing_index(narratives)

    assert dropped == 3
    assert all("Index" not in block.text for block in kept)
    assert len(kept) == 18


def test_mid_document_index_mention_not_truncated() -> None:
    narratives = [
        _block("Intro paragraph.", page_number=1),
        _block(
            "The consumer price index is used to adjust these brackets.",
            page_number=1,
        ),
        *[_block(f"Later body {i}.", page_number=2) for i in range(20)],
    ]

    kept, dropped = trim_trailing_index(narratives)

    assert dropped == 0
    assert len(kept) == len(narratives)


def test_title_index_mid_document_guarded() -> None:
    # An "Index" Title that appears early (low block index, early page) must
    # NOT trigger truncation in a long, multi-page document.
    narratives = [
        _block("Index", element_type="Title", page_number=1),
        *[_block(f"Body {i}.", page_number=2 + i // 4) for i in range(30)],
    ]

    kept, dropped = trim_trailing_index(narratives)

    assert dropped == 0
    assert len(kept) == len(narratives)


def test_all_junk_yields_empty() -> None:
    narratives = [
        _block("Page 1 of 4"),
        _block("Userid: CPM"),
        _block("Draft"),
    ]

    kept, dropped = trim_leading_irs_junk(narratives)

    assert kept == ()
    assert dropped == 3
    assert filter_irs_narratives(narratives) == ()


def test_empty_input_safe() -> None:
    assert filter_irs_narratives([]) == ()
    assert trim_leading_irs_junk([]) == ((), 0)
    assert trim_trailing_index([]) == ((), 0)


def test_uncategorized_junk_dropped_real_content_kept() -> None:
    narratives = [
        _block("Instructions for Form 1040-C", element_type="Title"),
        _block("Draft", element_type="UncategorizedText"),
        _block(
            "Schedule A lists itemized deductions.",
            element_type="UncategorizedText",
        ),
    ]

    filtered = filter_irs_narratives(narratives)

    assert [b.text for b in filtered] == [
        "Instructions for Form 1040-C",
        "Schedule A lists itemized deductions.",
    ]


def test_image_and_figure_blocks_dropped() -> None:
    narratives = [
        _block("Instructions for Form 1040-C", element_type="Title"),
        _block("Diagram of the 1040 workflow.", element_type="Image"),
        _block("Figure showing IRS Logo.", element_type="Figure"),
        _block("Enter your wages on line 1.", element_type="NarrativeText"),
    ]

    filtered = filter_irs_narratives(narratives)

    assert [b.element_type for b in filtered] == ["Title", "NarrativeText"]


def test_disabled_settings_passthrough() -> None:
    narratives = [
        _block("Page 1 of 12"),
        _block("Userid: CPM"),
        _block("Instructions for Form 1040-C", element_type="Title"),
    ]

    filtered = filter_irs_narratives(narratives, settings=_settings(enabled=False))

    assert filtered == tuple(narratives)


def test_enabled_settings_filters() -> None:
    narratives = [
        _block("Page 1 of 12"),
        _block("Instructions for Form 1040-C", element_type="Title"),
    ]

    filtered = filter_irs_narratives(narratives, settings=_settings(enabled=True))

    assert len(filtered) == 1
    assert filtered[0].text == "Instructions for Form 1040-C"
