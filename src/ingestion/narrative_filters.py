"""Deterministic Layer 1 content hygiene for parsed IRS narratives.

IRS Forms/Instructions/Publications routinely carry two classes of
non-substantive text that pollute retrieval embeddings and parent context:

* **Leading print/production metadata** on page 1 (proof banners, XSL/XML
  ``Fileid`` lines, ``Userid:``/``Schema:`` markers, ``Ok to Print`` /
  ``Draft`` stamps, production timestamps, ``Page N of M`` footers).
* **Trailing index sections** that begin with an ``Index`` title and run to
  the end of the document.

This module removes both with pure, deterministic regex rules (no LLM calls)
*before* parent/child chunking.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from core.logging_config import get_logger

if TYPE_CHECKING:
    from core.config import Settings
    from ingestion.unstructured_parser import NarrativeBlock

logger = get_logger(__name__)

# Trailing-index guard: an ``Index`` title only triggers truncation when it
# falls within this fraction of the document tail.
_TRAILING_COUNT_FRACTION = 0.10
_TRAILING_PAGE_FRACTION = 0.20

# Curated IRS print/proof metadata signatures (case-insensitive). Seeded from
# real IRS instruction PDFs (e.g. Instructions for Form 1040-C).
_LEADING_JUNK_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
        r"The type and rule above prints on all proofs",
        r"MUST be removed before printing",
        r"XSL/XML\s+Fileid",
        r"^\s*Userid:\s*",
        r"^\s*Schema:\s*$",
        r"^\s*instrx\s*$",
        r"^\s*Ok to Print\s*$",
        r"^\s*Draft\s*$",
        r"Leadpct:",
        r"Pt\.\s*size:",
        r"\(Init\.\s*&\s*Date\)",
        r"^\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}-[A-Za-z]{3,}-\d{4}\s*$",
    )
)

_INDEX_TITLE_RE = re.compile(r"^index\b", re.IGNORECASE)

_WHITESPACE_RE = re.compile(r"\s+")


def _is_junk_text(text: str) -> bool:
    """Return True if ``text`` matches any IRS print/proof metadata signature."""

    return any(pattern.search(text) for pattern in _LEADING_JUNK_PATTERNS)


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def trim_leading_irs_junk(
    narratives: Sequence[NarrativeBlock],
) -> tuple[tuple[NarrativeBlock, ...], int]:
    """Drop leading IRS print/proof metadata blocks.

    Returns the surviving blocks plus the number of leading blocks removed.
    Stops at the first block that does not match the junk denylist.
    """

    start_idx = 0
    for block in narratives:
        if _is_junk_text(block.text):
            start_idx += 1
            continue
        break
    return tuple(narratives[start_idx:]), start_idx


def trim_trailing_index(
    narratives: Sequence[NarrativeBlock],
) -> tuple[tuple[NarrativeBlock, ...], int]:
    """Truncate a trailing ``Index`` section and everything after it.

    Only fires when the index title sits inside the document tail (guard),
    avoiding false-positives from mid-document mentions of "index".
    Returns the surviving blocks plus the number of trailing blocks removed.
    """

    total = len(narratives)
    if total == 0:
        return (), 0

    max_page = max(
        (block.page_number for block in narratives if block.page_number is not None),
        default=None,
    )

    for idx, block in enumerate(narratives):
        if block.element_type.lower() != "title":
            continue
        if not _INDEX_TITLE_RE.match(_normalize(block.text)):
            continue
        if _is_in_trailing_zone(idx, total, block.page_number, max_page):
            return tuple(narratives[:idx]), total - idx

    return tuple(narratives), 0


def _is_in_trailing_zone(
    idx: int,
    total: int,
    page_number: int | None,
    max_page: int | None,
) -> bool:
    """True when block ``idx`` is in the last 10% by count or last 20% by page."""

    if idx >= total * (1.0 - _TRAILING_COUNT_FRACTION):
        return True
    return (
        page_number is not None
        and max_page is not None
        and max_page > 0
        and page_number >= max_page * (1.0 - _TRAILING_PAGE_FRACTION)
    )


_IMAGE_ELEMENT_TYPES: frozenset[str] = frozenset({"image", "figure"})


def _drop_image_blocks(
    narratives: Sequence[NarrativeBlock],
) -> tuple[NarrativeBlock, ...]:
    """Drop pure image/figure element blocks."""

    return tuple(
        block
        for block in narratives
        if block.element_type.lower() not in _IMAGE_ELEMENT_TYPES
    )


def _drop_uncategorized_junk(
    narratives: Sequence[NarrativeBlock],
) -> tuple[NarrativeBlock, ...]:
    """Drop ``UncategorizedText`` blocks that also match a junk signature."""

    return tuple(
        block
        for block in narratives
        if not (
            block.element_type.lower() == "uncategorizedtext"
            and _is_junk_text(block.text)
        )
    )


def filter_irs_narratives(
    narratives: Sequence[NarrativeBlock],
    *,
    settings: Settings | None = None,
) -> tuple[NarrativeBlock, ...]:
    """Apply Layer 1 narrative hygiene in order: leading, trailing, type.

    When ``settings.narrative_content_filter_enabled`` is False the input is
    returned unchanged, matching pre-filter behaviour.
    """

    if settings is not None and not settings.narrative_content_filter_enabled:
        return tuple(narratives)

    before = len(narratives)
    after_leading, leading_dropped = trim_leading_irs_junk(narratives)
    after_trailing, trailing_dropped = trim_trailing_index(after_leading)
    after_images = _drop_image_blocks(after_trailing)
    filtered = _drop_uncategorized_junk(after_images)

    logger.info(
        "narrative_content_filtered",
        narratives_before=before,
        narratives_after=len(filtered),
        leading_dropped=leading_dropped,
        trailing_dropped=trailing_dropped,
        images_dropped=len(after_trailing) - len(after_images),
    )
    return filtered
