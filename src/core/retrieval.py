"""Parent-child hybrid retrieval pipeline.

Pipeline order:

1. **Query router**: :func:`~core.query_router.route_query` makes a single cheap
   LLM call that simultaneously enforces the domain gate (raises
   :class:`~core.errors.OutOfDomainQueryError` for off-topic queries) and
   returns structured Qdrant filter hints (``tax_year``, ``doc_type``,
   ``form_numbers``).  Router failures fall back to unfiltered retrieval.
2. **Hybrid stage**: Qdrant ``query_points`` with a dense (cosine) prefetch
   and a sparse (BM25) prefetch, fused via Reciprocal Rank Fusion (RRF).
3. **Filter relaxation**: if the filtered search returns no hits, retry once
   with no filters.
4. **Layer 2 confidence gate**: reject weak/ambiguous retrievals based on the
   top hit's dense cosine similarity.
5. **Parent expansion**: walk the RRF-ordered child list and collect unique
   parent nodes up to ``retrieval_top_k_parents``.  Children that share a parent
   naturally collapse, so the top-ranked parent is whichever one contains the
   highest-RRF child.
6. **Context assembly**: gather matched child IDs and source URLs for the
   selected parents.

The retrieved context is what the answer-generation stage actually sees.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Real
from uuid import UUID

from pydantic import HttpUrl
from qdrant_client.models import SparseVector

from core.config import Settings, get_settings
from core.errors import OUT_OF_DOMAIN_MESSAGE, OutOfDomainQueryError, RetrievalError, RouterError
from core.logging_config import get_logger
from core.models import ParentNode, RetrievedContext
from core.query_router import QueryRouteResult, RouteFilters, route_query
from core.repository import DocumentRepository
from core.security import SanitizedQuery
from core.vector_store import HybridSearchResult, QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder
from ingestion.sparse_encoder import SparseEncoder

logger = get_logger(__name__)


def _normalize_rows(
    results: list[HybridSearchResult],
) -> list[dict[str, object]]:
    """Convert Qdrant search results into retrieval row dicts for the gate."""
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
    """Dense + BM25 hybrid retriever with LLM routing and parent expansion."""

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

    async def retrieve(
        self,
        query: str,
        *,
        sanitized: SanitizedQuery | None = None,
    ) -> RetrievedContext:
        """Run the full hybrid retrieval pipeline for a single query."""

        if not query or not query.strip():
            raise RetrievalError("Empty query")

        top_k = self._settings.retrieval_top_k_children

        if sanitized is not None:
            san = sanitized
        else:
            start = self._settings.user_query_start_tag
            end = self._settings.user_query_end_tag
            san = SanitizedQuery(
                cleaned_text=query,
                fenced_prompt_section=f"[{start}]\n{query}\n[{end}]",
                start_tag=start,
                end_tag=end,
            )

        route: QueryRouteResult | None = None
        try:
            route = await route_query(san, settings=self._settings)
        except OutOfDomainQueryError:
            raise
        except RouterError as exc:
            logger.warning(
                "query_router_fallback",
                error=str(exc),
                query_preview=query[:120],
            )
            route = None

        filters: RouteFilters = route.filters if route else RouteFilters()

        dense_embedding, sparse_vector = await _encode_query(
            query, self._embedder, self._sparse_encoder
        )

        has_filters = bool(
            filters.tax_year is not None
            or filters.doc_type is not None
            or filters.form_numbers
        )

        results = await self._vector_store.hybrid_search(
            dense_vector=dense_embedding,
            sparse_vector=sparse_vector,
            top_k=top_k,
            tax_year=filters.tax_year,
            doc_type=filters.doc_type,
            form_numbers=filters.form_numbers,
        )

        if not results and has_filters:
            logger.info(
                "retrieval_filter_relaxation",
                tax_year=filters.tax_year,
                doc_type=filters.doc_type,
                form_numbers=filters.form_numbers,
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

        top_k_parents = self._settings.retrieval_top_k_parents
        unique_parent_ids: list[UUID] = list(
            dict.fromkeys(UUID(str(r["parent_id"])) for r in rows)
        )[:top_k_parents]

        parent_records = await self._repo.fetch_parents(unique_parent_ids)

        selected_parents: OrderedDict[UUID, ParentNode] = OrderedDict()
        for pid in unique_parent_ids:
            record = parent_records.get(pid)
            if record is None:
                continue
            parent_meta: dict[str, object] = record["metadata"]
            selected_parents[pid] = ParentNode(
                id=pid,
                doc_id=UUID(str(record["doc_id"])),
                text_content=str(record["text_content"]),
                metadata=parent_meta,
            )

        # Collect all matched child IDs per selected parent.
        child_ids_by_parent: dict[UUID, list[UUID]] = {
            pid: [] for pid in selected_parents
        }
        for row in rows:
            pid = UUID(str(row["parent_id"]))
            cid = UUID(str(row["child_id"]))
            if pid in child_ids_by_parent:
                child_ids_by_parent[pid].append(cid)

        matched_child_ids: list[UUID] = []
        source_urls: list[HttpUrl] = []
        seen_source_urls: set[str] = set()

        for parent_id, parent_node in selected_parents.items():
            matched_child_ids.extend(child_ids_by_parent.get(parent_id, []))
            url = parent_node.metadata.get("source_url")
            if isinstance(url, str) and url and url not in seen_source_urls:
                seen_source_urls.add(url)
                source_urls.append(HttpUrl(url))

        return RetrievedContext(
            query=query,
            parent_nodes=tuple(selected_parents.values()),
            matched_child_ids=tuple(matched_child_ids),
            source_urls=tuple(source_urls),
        )


async def _encode_query(
    query: str,
    embedder: HuggingFaceEmbedder,
    sparse_encoder: SparseEncoder,
) -> tuple[tuple[float, ...], SparseVector]:
    """Concurrently encode the query for both dense and sparse retrieval."""
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

    This is the Layer 2 hybrid-score gate.  It runs after retrieval and before
    the answer LLM, complementing the router's Layer 1 domain check.
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
