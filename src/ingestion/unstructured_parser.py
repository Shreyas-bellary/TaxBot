from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, Tag
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared

from core.config import Settings, get_settings
from core.errors import UnstructuredParseError, UnsupportedPDFError
from core.logging_config import get_logger
from ingestion.narrative_filters import filter_irs_narratives

_ADOBE_STUB_PHRASES: tuple[str, ...] = (
    "requires Adobe Reader",
    "Adobe Reader installed",
    "pdf_forms_configure",
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class NarrativeBlock:
    """A semantic text block extracted from a PDF."""

    element_id: str
    element_type: str
    text: str
    page_number: int | None
    section: str | None


@dataclass(frozen=True, slots=True)
class TableBlock:
    """A table element with both raw and markdown payloads."""

    element_id: str
    text: str
    html: str
    markdown: str
    page_number: int | None
    section: str | None


@dataclass(frozen=True, slots=True)
class UnstructuredDocument:
    """The structured output we hand back to the ingestion pipeline."""

    narratives: tuple[NarrativeBlock, ...] = field(default_factory=tuple)
    tables: tuple[TableBlock, ...] = field(default_factory=tuple)


def _is_adobe_reader_stub(document: "UnstructuredDocument", filename: str) -> bool:
    """Return True if the parsed content looks like an Adobe Reader stub page.
    """
    first_page = [
        b for b in document.narratives if b.page_number in (None, 1)
    ]
    combined = " ".join(b.text for b in first_page[:20])
    return any(phrase in combined for phrase in _ADOBE_STUB_PHRASES)


class UnstructuredParser:
    """High-level entry point used by ingestion code."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = UnstructuredClient(
            api_key_auth=self._settings.unstructured_api_key.get_secret_value(),
            server_url=str(self._settings.unstructured_api_url),
        )

    async def partition(self, *, filename: str, content: bytes) -> UnstructuredDocument:
        """Submit a PDF for ``hi_res`` partitioning."""

        request = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(content=content, file_name=filename),
                strategy=shared.Strategy.HI_RES,
                pdf_infer_table_structure=True,
                languages=["eng"],
            ),
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.irs_max_retries + 1),
            wait=wait_exponential(multiplier=1.0, max=20.0),
            retry=retry_if_exception_type(UnstructuredParseError),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await self._client.general.partition_async(request=request)
                except Exception as exc:  # pragma: no cover - network surface
                    raise UnstructuredParseError(
                        f"Unstructured API call failed for {filename}: {exc}"
                    ) from exc

                elements = getattr(response, "elements", None)
                if not elements:
                    raise UnstructuredParseError(f"No elements returned for {filename}")
                document = _elements_to_document(elements)
                if _is_adobe_reader_stub(document, filename):
                    raise UnsupportedPDFError(
                        f"{filename} is an XFA/AcroForm stub that requires Adobe Reader "
                        "and contains no parseable content — skipping."
                    )
                filtered_narratives = filter_irs_narratives(
                    document.narratives,
                    settings=self._settings,
                )
                return UnstructuredDocument(
                    narratives=filtered_narratives,
                    tables=document.tables,
                )
        raise UnstructuredParseError(f"Unstructured partition exhausted retries for {filename}")


def _elements_to_document(elements: list[Any]) -> UnstructuredDocument:
    """Coerce the raw Unstructured payload into our typed model."""

    narratives: list[NarrativeBlock] = []
    tables: list[TableBlock] = []

    for raw in elements:
        element = raw if isinstance(raw, dict) else getattr(raw, "model_dump", lambda: {})()
        if not element:
            continue

        element_type = str(element.get("type", "")).strip()
        element_id = str(element.get("element_id") or element.get("id") or "")
        text = str(element.get("text") or "").strip()
        metadata = element.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        page_number = metadata.get("page_number")
        section = metadata.get("section")
        html = str(metadata.get("text_as_html") or "")

        if element_type.lower() == "table":
            markdown = html_table_to_markdown(html) if html else text
            tables.append(
                TableBlock(
                    element_id=element_id,
                    text=text,
                    html=html,
                    markdown=markdown,
                    page_number=int(page_number) if isinstance(page_number, int) else None,
                    section=str(section) if section else None,
                )
            )
            continue

        if not text:
            continue

        narratives.append(
            NarrativeBlock(
                element_id=element_id,
                element_type=element_type or "NarrativeText",
                text=text,
                page_number=int(page_number) if isinstance(page_number, int) else None,
                section=str(section) if section else None,
            )
        )

    return UnstructuredDocument(
        narratives=tuple(narratives),
        tables=tuple(tables),
    )


def html_table_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    found = soup.find("table")
    table: Tag = found if isinstance(found, Tag) else soup
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if not isinstance(tr, Tag):
            continue
        cells: list[str] = []
        for cell in tr.find_all(["th", "td"]):
            if not isinstance(cell, Tag):
                continue
            cell_text = " ".join(cell.get_text(separator=" ", strip=True).split())
            cells.append(cell_text.replace("|", "\\|"))
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    normalised = [row + [""] * (width - len(row)) for row in rows]
    header, *body = normalised
    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
