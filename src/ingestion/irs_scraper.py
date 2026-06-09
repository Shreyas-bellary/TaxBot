from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, get_settings
from core.errors import IngestionError
from core.logging_config import get_logger
from core.models import IRSDocumentMetadata

logger = get_logger(__name__)

IRS_AJAX_URL = "https://www.irs.gov/views/ajax"
IRS_HOST_ROOT = "https://www.irs.gov"

_DEFAULT_PARAMS: dict[str, str] = {
    "view_name": "pup_picklists",
    "view_display_id": "forms_and_pubs",
    "view_args": "",
    "view_path": "/node/16759",
    "view_base_path": "",
    "view_dom_id": "irs-forms-and-pubs",
    "pager_element": "0",
    "items_per_page": "200",
}

# Sort options exposed by the Views API. ``revision_date_desc`` is used by the
# nightly delta job to fetch freshly revised documents first and short-circuit
# the page loop the moment we encounter records we already have.
_SORT_PARAMS: dict[str, dict[str, str]] = {
    "natural": {},
    "revision_date_desc": {
        "order": "picklist_revision_date_iso",
        "sort": "desc",
    },
    "posted_date_desc": {
        "order": "posted_date",
        "sort": "desc",
    },
}

_VERSION_TITLE_RE = re.compile(r"\bversion\b", re.IGNORECASE)


class IRSAJAXClient:
    """Async client that walks the IRS Drupal Views AJAX pager.

    Parameters
    ----------
    settings:
        Loaded :class:`Settings`. Defaults to :func:`get_settings`.
    transport:
        Optional preconfigured httpx transport (used by tests via ``respx``).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            timeout=self._settings.irs_request_timeout_seconds,
            headers={
                "User-Agent": "TaxBot/1.0 (+https://taxbot.local)",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            },
            transport=transport,
        )

    async def __aenter__(self) -> IRSAJAXClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def iter_documents(
        self,
        *,
        sort: str = "natural",
        drop_multilingual: bool = True,
        max_pages: int | None = None,
    ) -> AsyncIterator[IRSDocumentMetadata]:
        """Yield every validated row across all pages.

        Pagination terminates when a page returns zero accepted rows.
        """

        if sort not in _SORT_PARAMS:
            raise ValueError(f"Unknown sort key: {sort!r}")

        page = 0
        while True:
            if max_pages is not None and page >= max_pages:
                logger.info("scraper_page_limit_reached", page=page)
                return

            html = await self._fetch_page_html(page=page, sort=sort)
            rows = list(self._parse_rows(html))
            logger.info(
                "scraper_page_fetched",
                page=page,
                row_count=len(rows),
                sort=sort,
            )
            if not rows:
                return

            yielded = 0
            for metadata in rows:
                if drop_multilingual and _VERSION_TITLE_RE.search(metadata.doc_title):
                    logger.debug(
                        "scraper_dropped_multilingual",
                        doc_number=metadata.doc_number,
                        doc_title=metadata.doc_title,
                    )
                    continue
                yielded += 1
                yield metadata

            logger.info(
                "scraper_page_yielded",
                page=page,
                yielded=yielded,
                filtered=len(rows) - yielded,
            )

            page += 1
            await asyncio.sleep(self._settings.irs_request_throttle_seconds)

    async def _fetch_page_html(self, *, page: int, sort: str) -> str:
        """Fetch a single page and return the decoded HTML payload."""

        params: dict[str, str] = {**_DEFAULT_PARAMS, **_SORT_PARAMS[sort]}
        params["pager_element"] = str(page)
        params["page"] = str(page)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.irs_max_retries + 1),
            wait=wait_exponential(multiplier=1.0, max=15.0),
            retry=retry_if_exception_type(
                (httpx.TransportError, httpx.HTTPStatusError, IngestionError)
            ),
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(IRS_AJAX_URL, params=params)
                response.raise_for_status()
                try:
                    payload: list[dict[str, Any]] | dict[str, Any] = response.json()
                except json.JSONDecodeError as exc:
                    raise IngestionError(
                        f"IRS AJAX page {page} returned non-JSON payload"
                    ) from exc
                return _extract_insert_html(payload, page=page)
        raise IngestionError(f"IRS AJAX page {page} exhausted retries")  # pragma: no cover

    @staticmethod
    def _parse_rows(html: str) -> Iterable[IRSDocumentMetadata]:
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.find_all("tr"):
            if not isinstance(tr, Tag):
                continue
            cells = tr.find_all("td", recursive=False)
            if len(cells) != 4:
                continue
            try:
                yield _row_to_metadata(cells)
            except (ValidationError, ValueError) as exc:
                logger.debug("scraper_row_rejected", reason=str(exc))


def _extract_insert_html(
    payload: list[dict[str, Any]] | dict[str, Any],
    *,
    page: int,
) -> str:
    """Locate the ``"insert"`` command inside a Drupal Views AJAX response."""

    commands: list[dict[str, Any]]
    if isinstance(payload, list):
        commands = payload
    elif isinstance(payload, dict) and isinstance(payload.get("commands"), list):
        commands = payload["commands"]
    else:
        raise IngestionError(f"IRS AJAX page {page} returned unsupported payload shape")

    insert_blocks = [
        item
        for item in commands
        if isinstance(item, dict) and item.get("command") == "insert"
    ]
    if not insert_blocks:
        return ""

    parts: list[str] = []
    for block in insert_blocks:
        data = block.get("data")
        if isinstance(data, str):
            parts.append(data)
    return "".join(parts)


def _row_to_metadata(cells: list[Tag]) -> IRSDocumentMetadata:
    """Convert exactly 4 ``<td>`` cells into a validated metadata model."""

    number_cell, title_cell, revision_cell, posted_cell = cells

    anchor = number_cell.find("a")
    if not isinstance(anchor, Tag) or not anchor.get("href"):
        raise ValueError("doc_number cell missing <a href>")

    href = str(anchor["href"]).strip()
    if href.startswith("http://") or href.startswith("https://"):
        pdf_url = href
    else:
        pdf_url = f"{IRS_HOST_ROOT}{href}" if href.startswith("/") else f"{IRS_HOST_ROOT}/{href}"

    return IRSDocumentMetadata(
        doc_number=_clean_text(number_cell.get_text()),
        doc_title=_clean_text(title_cell.get_text()),
        revision_date=_clean_text(revision_cell.get_text()),
        posted_date=_clean_text(posted_cell.get_text()),
        pdf_url=pdf_url,  # type: ignore[arg-type]
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
