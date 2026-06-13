"""Reusable filter predicates applied to scraped IRS rows.

Two predicates currently live here:

* :func:`is_within_backfill_window` enforces the *last-5-tax-years* rule.
* :func:`is_language_allowed` enforces the language allowlist.
* :func:`reject_oversized_publication` skips long publications before parsing.
"""

from __future__ import annotations

from datetime import datetime

from core.config import Settings
from core.errors import OversizedPublicationError
from core.models import DocCategory, IRSDocumentMetadata
from ingestion.pdf_fetcher import count_pdf_pages


def is_within_backfill_window(
    metadata: IRSDocumentMetadata,
    *,
    settings: Settings,
    now: datetime | None = None,
) -> bool:
    """Return ``True`` if the document is recent enough for backfill.

    A document is considered in-window if its inferred ``tax_year`` is
    greater than or equal to :attr:`Settings.backfill_oldest_tax_year`, or
    if no year can be inferred at all (older "evergreen" Publications are
    re-published infrequently so we err on the side of inclusion when the
    year is unknown).
    """

    del now  # currently unused, reserved for future date-anchored gates
    tax_year = metadata.tax_year
    if tax_year is None:
        return True
    return tax_year >= settings.backfill_oldest_tax_year


def is_language_allowed(metadata: IRSDocumentMetadata, *, settings: Settings) -> bool:
    """Currently the scraper strips multilingual rows by title.

    This helper is a future-proof hook: when the allowlist is widened to
    include other languages, the scraper can be reconfigured to keep
    ``(Spanish version)`` etc., and this predicate will gate them on the
    permitted set.
    """

    if settings.multilingual_enabled and settings.language_allowlist == "*":
        return True
    allowed = settings.language_tags
    title_lower = metadata.doc_title.lower()
    if "english" in title_lower or "(en)" in title_lower or "version" not in title_lower:
        return "en" in allowed
    return any(tag != "en" and tag in title_lower for tag in allowed)


def reject_oversized_publication(
    metadata: IRSDocumentMetadata,
    pdf_content: bytes,
    *,
    settings: Settings,
) -> None:
    """Raise if a publication exceeds :attr:`Settings.publication_max_pages`.

    A limit of ``0`` disables this check.
    """

    max_pages = settings.publication_max_pages
    if max_pages <= 0 or metadata.category != DocCategory.PUBLICATION:
        return

    page_count = count_pdf_pages(pdf_content)
    if page_count is None or page_count <= max_pages:
        return

    raise OversizedPublicationError(
        f"{metadata.doc_number} has {page_count} pages (limit {max_pages}) — skipping."
    )
