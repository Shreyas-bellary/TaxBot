"""Async PDF downloader with content hashing.

The fetcher streams each PDF to memory (IRS PDFs are typically <2 MiB) and
returns the bytes alongside a SHA-256 digest. The digest is the canonical
input to the metadata-or-hash delta check in
:mod:`ingestion.delta_update`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, get_settings
from core.errors import IngestionError
from core.logging_config import get_logger

logger = get_logger(__name__)


def count_pdf_pages(content: bytes) -> int | None:
    """Return the page count for a PDF payload, or ``None`` when unreadable."""

    try:
        reader = PdfReader(BytesIO(content), strict=False)
        return len(reader.pages)
    except PdfReadError:
        return None

@dataclass(frozen=True, slots=True)
class FetchedPDF:
    """Result of an IRS PDF download."""

    url: str
    content: bytes
    sha256: str
    content_type: str


class PDFFetcher:
    """Async fetcher with bounded retries and a per-request timeout."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = httpx.AsyncClient(
            timeout=self._settings.irs_request_timeout_seconds,
            headers={"User-Agent": "TaxBot/1.0 (+https://taxbot.local)"},
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> PDFFetcher:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(self, url: str) -> FetchedPDF:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.irs_max_retries + 1),
            wait=wait_exponential(multiplier=1.0, max=15.0),
            retry=retry_if_exception_type(httpx.TransportError),
            reraise=True,
        ):
            with attempt:
                response = await self._client.get(url)
                response.raise_for_status()
                payload = response.content
                if not payload:
                    raise IngestionError(f"Empty PDF payload from {url}")
                digest = hashlib.sha256(payload).hexdigest()
                logger.info(
                    "pdf_fetched",
                    url=url,
                    bytes=len(payload),
                    sha256=digest,
                )
                return FetchedPDF(
                    url=url,
                    content=payload,
                    sha256=digest,
                    content_type=response.headers.get("content-type", "application/pdf"),
                )
        raise IngestionError(f"Exhausted retries fetching {url}")  # pragma: no cover
