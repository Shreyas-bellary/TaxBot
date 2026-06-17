"""Parent-child hybrid retrieval pipeline.

Pipeline order:

1. **Pre-filter**: extract structured metadata hints (``tax_year``,
   ``form_number``, ``doc_type``) from the user query, then push them as
   Qdrant payload filter conditions.
2. **Hybrid stage**: Qdrant ``query_points`` with a dense (cosine) prefetch
   and a sparse (BM25) prefetch, fused via Reciprocal Rank Fusion (RRF).
3. **Layer 2 confidence gate**: reject weak/ambiguous retrievals based on the
   top hit's dense cosine similarity (which is a 0-1 absolute relevance
   measure, unlike the rank-based RRF score).
4. **Parent expansion**: look up parent rows by primary key from Postgres,
   deduplicate, and assemble a :class:`RetrievedContext` payload.

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
from qdrant_client.models import SparseVector

from core.config import Settings, get_settings
from core.errors import OUT_OF_DOMAIN_MESSAGE, OutOfDomainQueryError, RetrievalError
from core.logging_config import get_logger
from core.models import ParentNode, RetrievedContext
from core.repository import DocumentRepository
from core.vector_store import HybridSearchResult, QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder
from ingestion.sparse_encoder import SparseEncoder

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
    "notice": "notice",
}

# Procedural-intent verbs: user is asking *how to do* something
_PROCEDURAL_INTENT_RE = re.compile(
    r"\b(?:instruct|how\s+(?:to|do|does|can|should|would)|determine|complete|fill\s+out|calculate|compute|figure)\b",
    re.IGNORECASE,
)

_INSTRUCTION_DOC_PREFIX = "Instruction"

@dataclass(frozen=True, slots=True)
class QueryFilters:
    """Structured filters extracted from a raw user query."""

    tax_year: int | None
    form_number: str | None
    doc_type: str | None
    form_number_variants: tuple[str, ...] = ()
    procedural_intent: bool = False


def _form_family_variants(prefix: str, body: str) -> tuple[str, ...]:
    """Build Qdrant MatchAny labels for a Form/Schedule and its instruction PDF.
    """
    form_number = f"{prefix} {body}".strip()
    if prefix in ("Form", "Schedule"):
        return (form_number, f"{_INSTRUCTION_DOC_PREFIX} {body}")
    return (form_number,)


def extract_filters(query: str) -> QueryFilters:
    """Best-effort extraction of metadata pre-filters from a natural query.
    """

    tax_year_match = _TAX_YEAR_RE.search(query)
    tax_year = int(tax_year_match.group(1)) if tax_year_match else None

    form_number: str | None = None
    form_number_variants: tuple[str, ...] = ()
    form_match = _FORM_NUMBER_RE.search(query)
    if form_match:
        prefix = form_match.group(0).split()[0].title()
        body = form_match.group(1).strip().upper()
        form_number = f"{prefix} {body}".strip()
        form_number_variants = _form_family_variants(prefix, body)

    lowered = query.lower()
    doc_type: str | None = None
    for hint, mapped in _DOC_TYPE_HINTS.items():
        if hint in lowered:
            doc_type = mapped
            break

    procedural_intent = bool(_PROCEDURAL_INTENT_RE.search(query))

    return QueryFilters(
        tax_year=tax_year,
        form_number=form_number,
        doc_type=doc_type,
        form_number_variants=form_number_variants,
        procedural_intent=procedural_intent,
    )


def _normalize_rows(
    results: list[HybridSearchResult],
) -> list[dict[str, object]]:
    """Convert Qdrant search results into retrieval row dicts for the gate.
    """
    if not results:
        return []

    dense_top_score = results[0].dense_top_score
    top_rrf = results[0].rrf_score

    rows: list[dict[str, object]] = []
    for hit in results:
        hybrid_score = (
            dense_top_score * (hit.rrf_score / top_rrf) if top_rrf > 0 else 0.0
        )
        rows.append(
            {
                "child_id": hit.child_id,
                "parent_id": hit.parent_id,
                "doc_id": hit.doc_id,
                "hybrid_score": hybrid_score,
            }
        )
    return rows


class HybridRetriever:
    """Dense + BM25 hybrid retriever with Qdrant and parent expansion."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedder: HuggingFaceEmbedder,
        vector_store: QdrantVectorStore,
        sparse_encoder: SparseEncoder,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_encoder = sparse_encoder
        self._settings = settings or get_settings()

    async def retrieve(self, query: str) -> RetrievedContext:
        """Run the hybrid retrieval pipeline for a single query."""

        if not query or not query.strip():
            raise RetrievalError("Empty query")

        filters = extract_filters(query)
        top_k = self._settings.retrieval_top_k_children

        dense_embedding, sparse_vector = await _encode_query(
            query, self._embedder, self._sparse_encoder
        )

        results: list[HybridSearchResult] = []
        if (
            filters.procedural_intent
            and filters.form_number_variants
            and filters.doc_type is None
        ):
            instruction_variant = next(
                (
                    v
                    for v in filters.form_number_variants
                    if v.startswith(f"{_INSTRUCTION_DOC_PREFIX} ")
                ),
                None,
            )
            if instruction_variant:
                results = await self._vector_store.hybrid_search(
                    dense_vector=dense_embedding,
                    sparse_vector=sparse_vector,
                    top_k=top_k,
                    tax_year=filters.tax_year,
                    form_number_variants=(instruction_variant,),
                    doc_type="instruction",
                )
                if results:
                    logger.info(
                        "retrieval_procedural_instruction_hit",
                        form_number=filters.form_number,
                        instruction_variant=instruction_variant,
                    )

        if not results:
            results = await self._vector_store.hybrid_search(
                dense_vector=dense_embedding,
                sparse_vector=sparse_vector,
                top_k=top_k,
                tax_year=filters.tax_year,
                form_number=filters.form_number,
                form_number_variants=filters.form_number_variants,
                doc_type=filters.doc_type,
            )

        # if filters yielded nothing, retry with no filters.
        if not results and (
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
            results = await self._vector_store.hybrid_search(
                dense_vector=dense_embedding,
                sparse_vector=sparse_vector,
                top_k=top_k,
            )

        if not results:
            raise RetrievalError("No matching child nodes for query")

        rows = _normalize_rows(results)
        assess_retrieval_confidence(
            rows,
            settings=self._settings,
            query_preview=query[:120],
        )

        # Fetch parent rows from Postgres in one round-trip.
        unique_parent_ids = list(
            dict.fromkeys(UUID(r["parent_id"]) for r in rows)  # type: ignore[arg-type]
        )
        parent_records = await self._repo.fetch_parents(unique_parent_ids)

        parents: OrderedDict[UUID, ParentNode] = OrderedDict()
        matched_child_ids: list[UUID] = []
        source_urls: list[HttpUrl] = []
        seen_source_urls: set[str] = set()
        top_k_parents = self._settings.retrieval_top_k_parents

        for row in rows:
            parent_id = UUID(str(row["parent_id"]))
            matched_child_ids.append(UUID(str(row["child_id"])))
            if parent_id not in parents:
                record = parent_records.get(parent_id)
                if record is None:
                    continue
                parent_metadata: dict[str, object] = record["metadata"]
                parents[parent_id] = ParentNode(
                    id=parent_id,
                    doc_id=UUID(str(record["doc_id"])),
                    text_content=str(record["text_content"]),
                    metadata=parent_metadata,
                )
                url = parent_metadata.get("source_url")
                if isinstance(url, str) and url and url not in seen_source_urls:
                    seen_source_urls.add(url)
                    source_urls.append(HttpUrl(url))
            if len(parents) >= top_k_parents:
                break

        return RetrievedContext(
            query=query,
            parent_nodes=tuple(parents.values()),
            matched_child_ids=tuple(matched_child_ids),
            source_urls=tuple(source_urls),
        )


async def _encode_query(
    query: str,
    embedder: HuggingFaceEmbedder,
    sparse_encoder: SparseEncoder,
) -> tuple[tuple[float, ...], SparseVector]:
    """Concurrently encode the query for both dense and sparse retrieval."""
    import asyncio

    dense_task = asyncio.ensure_future(embedder.embed(query))
    sparse_task = asyncio.ensure_future(sparse_encoder.embed_query(query))
    dense_embedding, sparse_vector = await asyncio.gather(dense_task, sparse_task)
    return dense_embedding, sparse_vector


def assess_retrieval_confidence(
    rows: Sequence[Mapping[str, object]],
    *,
    settings: Settings,
    query_preview: str,
) -> None:
    """Raise :class:`OutOfDomainQueryError` if retrieval confidence is weak.
    """
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
