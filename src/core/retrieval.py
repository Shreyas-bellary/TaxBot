"""Parent-child hybrid retrieval pipeline.

Pipeline order:

1. **Pre-filter**: extract structured metadata hints (``tax_year``,
   ``form_number``, ``doc_type``) from the user query, then push them as
   exact SQL ``WHERE`` clauses.
2. **Hybrid stage**: combine a vector cosine match against the
   ``child_nodes`` embeddings with a Postgres FTS keyword score. The hybrid
   ranking is ``0.75 * vector + 0.25 * fts`` and runs inside a single SQL
   statement to avoid round-trip overhead.
3. **Layer 2 confidence gate**: reject weak/ambiguous retrievals from actual
   ``hybrid_score`` values (no extra LLM call).
4. **Parent expansion**: deduplicate by parent, fetch the verbatim parent
   markdown, and assemble a :class:`RetrievedContext` payload.

The retrieved context is what the answer-generation stage actually sees.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from uuid import UUID

from pydantic import HttpUrl

from core.config import Settings, get_settings
from core.errors import OUT_OF_DOMAIN_MESSAGE, OutOfDomainQueryError, RetrievalError
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
            logger.info(
                "retrieval_filter_relaxation",
                tax_year=filters.tax_year,
                form_number=filters.form_number,
                doc_type=filters.doc_type,
            )
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
        assess_retrieval_confidence(
            rows,
            settings=self._settings,
            query_preview=query[:120],
        )

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


def assess_retrieval_confidence(
    rows: Sequence[Mapping[str, object]],
    *,
    settings: Settings,
    query_preview: str,
) -> None:
    """Raise :class:`OutOfDomainQueryError` if retrieval confidence is weak."""

    if not settings.retrieval_confidence_gate_enabled or not rows:
        return

    top_score = _coerce_hybrid_score(rows[0])
    threshold = settings.retrieval_min_hybrid_score
    if top_score < threshold:
        logger.warning(
            "retrieval_confidence_rejected",
            reason="top_score_below_threshold",
            top_score=top_score,
            threshold=threshold,
            query_preview=query_preview,
        )
        raise OutOfDomainQueryError(OUT_OF_DOMAIN_MESSAGE)

    gap_threshold = settings.retrieval_min_score_gap
    if gap_threshold is None or len(rows) < 2:
        return

    second_score = _coerce_hybrid_score(rows[1])
    score_gap = top_score - second_score
    if score_gap < gap_threshold:
        logger.warning(
            "retrieval_confidence_rejected",
            reason="top2_gap_below_threshold",
            top_score=top_score,
            second_score=second_score,
            score_gap=score_gap,
            gap_threshold=gap_threshold,
            query_preview=query_preview,
        )
        raise OutOfDomainQueryError(OUT_OF_DOMAIN_MESSAGE)


def _coerce_hybrid_score(row: Mapping[str, object]) -> float:
    value = row.get("hybrid_score")
    if isinstance(value, Real):
        return float(value)
    raise RetrievalError("Retrieval row missing numeric hybrid_score")
