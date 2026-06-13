"""Heuristics for dropping non-substantive table blocks during ingest.

Unstructured sometimes emits one-cell fragments, single words, or sentence
stubs as ``table`` elements. These cannot produce a valid three-sentence
summary and should not trigger fallback LLM calls or orphan parent nodes.
"""

from __future__ import annotations

import re

from core.errors import SummarizationError, TrivialTableError

_SEPARATOR_ROW_RE = re.compile(r"^\|\s*[-:\s|]+\|\s*$")


def _table_data_rows(markdown: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in markdown.strip().splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if _SEPARATOR_ROW_RE.match(stripped):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def is_trivial_table(markdown: str) -> bool:
    """Return True when markdown is too small to be a useful indexed table."""
    text = markdown.strip()
    if not text:
        return True

    rows = _table_data_rows(text)
    if not rows:
        # Plain text with no pipe structure — treat short blobs as trivial.
        return len(text.split()) <= 12

    total_words = sum(len(cell.split()) for row in rows for cell in row)
    column_count = max(len(row) for row in rows)

    # Header + at least one data row with multiple columns is a real table.
    if len(rows) >= 2 and column_count >= 2:
        return False

    if total_words < 6:
        return True

    return len(rows) == 1 and (column_count == 1 or total_words <= 4)


def is_summarization_validation_error(exc: BaseException) -> bool:
    """True when the summariser rejected LLM output shape, not a transport error."""
    if not isinstance(exc, SummarizationError):
        return False
    message = str(exc)
    return (
        "Summary sentence too short" in message
        or "Expected ~3 sentences" in message
        or "Empty summary text" in message
    )


def should_drop_table_after_summary_failure(
    markdown: str,
    exc: BaseException,
) -> bool:
    """Return True when a failed summary means the table parent should be dropped."""
    if isinstance(exc, TrivialTableError):
        return True
    if is_trivial_table(markdown):
        return True
    return is_summarization_validation_error(exc) and is_trivial_table(markdown)
