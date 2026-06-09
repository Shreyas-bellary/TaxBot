"""Parent-child hybrid retrieval pipeline.

Pipeline order:

1. **Pre-filter**: extract structured metadata hints (``tax_year``,
   ``form_number``, ``doc_type``) from the user query, then push them as
   exact SQL ``WHERE`` clauses.
2. **Hybrid stage**: combine a vector cosine match against the
   ``child_nodes`` embeddings with a Postgres FTS keyword score. The hybrid
   ranking is ``0.75 * vector + 0.25 * fts`` and runs inside a single SQL
   statement to avoid round-trip overhead.
3. **Parent expansion**: deduplicate by parent, fetch the verbatim parent
   markdown, and assemble a :class:`RetrievedContext` payload.

The retrieved context is what the answer-generation stage actually sees.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from uuid import UUID

from pydantic import HttpUrl

from core.config import Settings, get_settings
from core.errors import RetrievalError
from core.logging_config import get_logger
from core.models import ParentNode, RetrievedContext
from core.repository import DocumentRepository
from ingestion.embeddings import HuggingFaceEmbedder

logger = get_logger(__name__)

_TAX_YEAR_RE = re.compile(r"\b(?:tax\s+year\s+)?((?:19|20)\d{2})\b", re.IGNORECASE)
_FORM_NUMBER_RE = re.compile(
    r"\b(?:Form|Publication|Pub|Schedule)\s+([0-9A-Z][0-9A-Z\-]*(?:\s+[A-Z](?![a-z]))?)",
    re.IGNORECASE,
)
_DOC_TYPE_HINTS: dict[str, str] = {
    "instructions": "instruction",
    "instruction": "instruction",
    "publication": "publication",
    "pub ": "publication",
    "pub.": "publication",
    "form": "form",
    "schedule": "form",
    "notice": "notice",
}


@dataclass(frozen=True, slots=True)
class QueryFilters:
    """Structured filters extracted from a raw user query."""

    tax_year: int | None
    form_number: str | None
    doc_type: str | None


def extract_filters(query: str) -> QueryFilters:
    """Best-effort extraction of metadata pre-filters from a natural query."""

    tax_year_match = _TAX_YEAR_RE.search(query)
    tax_year = int(tax_year_match.group(1)) if tax_year_match else None

    form_number: str | None = None
    form_match = _FORM_NUMBER_RE.search(query)
    if form_match:
        prefix = form_match.group(0).split()[0].title()
        body = form_match.group(1).strip().upper()
        form_number = f"{prefix} {body}".strip()

    lowered = query.lower()
    doc_type: str | None = None
    for hint, mapped in _DOC_TYPE_HINTS.items():
        if hint in lowered:
            doc_type = mapped
            break

    return QueryFilters(
        tax_year=tax_year,
        form_number=form_number,
        doc_type=doc_type,
    )


class HybridRetriever:
    """Hybrid FTS + vector retriever with parent expansion."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedder: HuggingFaceEmbedder,
        *,
        settings: Settings | None = None,
        top_k_children: int = 24,
        top_k_parents: int = 6,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._settings = settings or get_settings()
        self._top_k_children = top_k_children
        self._top_k_parents = top_k_parents

    async def retrieve(self, query: str) -> RetrievedContext:
        """Run the hybrid retrieval pipeline for a single query."""

        if not query or not query.strip():
            raise RetrievalError("Empty query")

        filters = extract_filters(query)
        embedding = await self._embedder.embed(query)

        rows = await self._repo.hybrid_retrieve(
            query_text=query,
            query_embedding=embedding,
            top_k_children=self._top_k_children,
            tax_year=filters.tax_year,
            form_number=filters.form_number,
            doc_type=filters.doc_type,
        )

        # If a strict filter killed all matches, retry without it. This is
        # safe because deterministic filters are heuristic, not authoritative.
        if not rows and (
            filters.tax_year is not None
            or filters.form_number is not None
            or filters.doc_type is not None
        ):
            logger.info("retrieval_filter_relaxation", filters=filters.__dict__)
            rows = await self._repo.hybrid_retrieve(
                query_text=query,
                query_embedding=embedding,
                top_k_children=self._top_k_children,
                tax_year=None,
                form_number=None,
                doc_type=None,
            )

        if not rows:
            raise RetrievalError("No matching child nodes for query")

        parents: OrderedDict[UUID, ParentNode] = OrderedDict()
        matched_child_ids: list[UUID] = []
        source_urls: list[HttpUrl] = []
        seen_source_urls: set[str] = set()

        for row in rows:
            parent_id = UUID(str(row["parent_id"]))
            matched_child_ids.append(UUID(str(row["child_id"])))
            if parent_id not in parents:
                parent_metadata = row["parent_metadata"]
                parents[parent_id] = ParentNode(
                    id=parent_id,
                    doc_id=UUID(str(row["doc_id"])),
                    text_content=row["parent_text"],
                    metadata=parent_metadata,
                )
                url = parent_metadata.get("source_url")
                if isinstance(url, str) and url and url not in seen_source_urls:
                    seen_source_urls.add(url)
                    source_urls.append(HttpUrl(url))
            if len(parents) >= self._top_k_parents:
                break

        return RetrievedContext(
            query=query,
            parent_nodes=tuple(parents.values()),
            matched_child_ids=tuple(matched_child_ids),
            source_urls=tuple(source_urls),
        )
