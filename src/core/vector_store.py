"""Qdrant-backed vector store for dense + sparse (BM25) hybrid retrieval.

:class:`QdrantVectorStore` wraps an :class:`~qdrant_client.AsyncQdrantClient`
and provides the three operations the ingestion pipeline and retrieval layer
need:

- :meth:`ensure_collection` — create the collection (dense + sparse vectors)
  and payload indexes if they do not already exist.
- :meth:`upsert_points` — write a batch of child-node vectors with metadata
  payload.
- :meth:`hybrid_search` — prefetch dense + sparse candidates and fuse with
  RRF; also returns the top hit's raw dense cosine similarity for the
  confidence gate.
- :meth:`delete_by_doc_id` — delete all points whose payload ``doc_id``
  matches the given value (used before re-ingestion).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from core.config import Settings, get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)

_DENSE_NAME = "dense"
_SPARSE_NAME = "sparse"

_PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("doc_id", models.PayloadSchemaType.KEYWORD),
    ("form_number", models.PayloadSchemaType.KEYWORD),
    ("doc_type", models.PayloadSchemaType.KEYWORD),
    ("tax_year", models.PayloadSchemaType.INTEGER),
)


def _build_metadata_filter(
    *,
    tax_year: int | None,
    form_number: str | None,
    doc_type: str | None,
) -> models.Filter | None:
    """Convert metadata filter values into a Qdrant :class:`~qdrant_client.models.Filter`.

    Returns ``None`` when all three values are ``None`` (no filter applied).
    The fields must match the payload keys written during ingest.
    """
    conditions: list[models.Condition] = []
    if tax_year is not None:
        conditions.append(
            models.FieldCondition(
                key="tax_year",
                match=models.MatchValue(value=tax_year),
            )
        )
    if form_number is not None:
        conditions.append(
            models.FieldCondition(
                key="form_number",
                match=models.MatchValue(value=form_number),
            )
        )
    if doc_type is not None:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


class HybridSearchResult:
    """Lightweight result container returned by :meth:`QdrantVectorStore.hybrid_search`."""

    __slots__ = ("child_id", "dense_top_score", "doc_id", "parent_id", "rrf_score")

    def __init__(
        self,
        *,
        child_id: str,
        parent_id: str,
        doc_id: str,
        rrf_score: float,
        dense_top_score: float,
    ) -> None:
        self.child_id = child_id
        self.parent_id = parent_id
        self.doc_id = doc_id
        self.rrf_score = rrf_score
        self.dense_top_score = dense_top_score


class QdrantVectorStore:
    """Async Qdrant store with dense + sparse (BM25) hybrid search."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = AsyncQdrantClient(
            url=str(self._settings.qdrant_url),
            api_key=self._settings.qdrant_api_key.get_secret_value(),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection and payload indexes if it doesn't exist.

        Idempotent — safe to call on every startup.
        """
        collection_name = self._settings.qdrant_collection
        exists = await self._client.collection_exists(collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    _DENSE_NAME: models.VectorParams(
                        size=self._settings.embedding_dimension,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    _SPARSE_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            logger.info(
                "qdrant_collection_created",
                collection=collection_name,
                dense_dim=self._settings.embedding_dimension,
            )
        else:
            logger.info("qdrant_collection_ready", collection=collection_name)

        await self._ensure_payload_indexes(collection_name)

    async def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes required for filtered delete and search."""
        for field_name, field_schema in _PAYLOAD_INDEXES:
            try:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
                logger.info(
                    "qdrant_payload_index_created",
                    collection=collection_name,
                    field=field_name,
                )
            except Exception as exc:
                # Index already exists on re-runs / existing collections.
                if "already exists" in str(exc).lower():
                    logger.debug(
                        "qdrant_payload_index_exists",
                        collection=collection_name,
                        field=field_name,
                    )
                    continue
                raise

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Ingest helpers
    # ------------------------------------------------------------------

    async def upsert_points(
        self,
        *,
        child_ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[models.SparseVector],
        payloads: Sequence[dict[str, Any]],
        doc_id: str,
    ) -> None:
        """Upsert child-node points into Qdrant in bounded batches.

        ``child_ids``, ``dense_vectors``, ``sparse_vectors``, and ``payloads``
        must all have the same length. Points are sent in batches of
        ``qdrant_upsert_batch_size`` to avoid request-size timeouts on large
        docs.
        """
        points = [
            models.PointStruct(
                id=cid,
                vector={
                    _DENSE_NAME: list(dense),
                    _SPARSE_NAME: sparse,
                },
                payload=payload,
            )
            for cid, dense, sparse, payload in zip(
                child_ids, dense_vectors, sparse_vectors, payloads, strict=False
            )
        ]
        if not points:
            return

        batch_size = self._settings.qdrant_upsert_batch_size
        collection = self._settings.qdrant_collection
        total = len(points)
        for start in range(0, total, batch_size):
            batch = points[start : start + batch_size]
            await self._client.upsert(
                collection_name=collection,
                points=batch,
            )
            logger.info(
                "qdrant_upsert",
                batch_start=start,
                batch_count=len(batch),
                total=total,
                doc_id=doc_id,
                collection=collection,
            )

    async def delete_by_doc_id(self, doc_id: str | UUID) -> None:
        """Delete all points whose payload ``doc_id`` equals the given value."""
        doc_id_str = str(doc_id)
        await self._client.delete(
            collection_name=self._settings.qdrant_collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id_str),
                    )
                ]
            ),
        )
        logger.info(
            "qdrant_delete",
            doc_id=doc_id_str,
            collection=self._settings.qdrant_collection,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        *,
        dense_vector: Sequence[float],
        sparse_vector: models.SparseVector,
        top_k: int,
        tax_year: int | None = None,
        form_number: str | None = None,
        doc_type: str | None = None,
    ) -> list[HybridSearchResult]:
        """Run Qdrant hybrid search (dense cosine + BM25 sparse, RRF fusion).

        Returns at most ``top_k`` results ordered by RRF score descending.
        Each result carries ``dense_top_score`` — the top hit's cosine
        similarity from a separate single-result dense query — which callers
        use as the absolute relevance signal for the confidence gate.
        """
        query_filter = _build_metadata_filter(
            tax_year=tax_year,
            form_number=form_number,
            doc_type=doc_type,
        )
        filtered = query_filter is not None

        dense_vec = list(dense_vector)
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using=_DENSE_NAME,
                limit=top_k,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=_SPARSE_NAME,
                limit=top_k,
                filter=query_filter,
            ),
        ]

        rrf_response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        rrf_points = rrf_response.points

        # Separate dense-only query (limit=1) to obtain the top cosine score
        # for the confidence gate (RRF scores are rank-based, not 0-1).
        dense_top_score = 0.0
        if rrf_points:
            dense_response = await self._client.query_points(
                collection_name=self._settings.qdrant_collection,
                query=dense_vec,
                using=_DENSE_NAME,
                limit=1,
                query_filter=query_filter,
                with_payload=False,
            )
            if dense_response.points:
                dense_top_score = float(dense_response.points[0].score)

        results: list[HybridSearchResult] = []
        for pt in rrf_points:
            payload = pt.payload or {}
            results.append(
                HybridSearchResult(
                    child_id=str(pt.id),
                    parent_id=str(payload.get("parent_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    rrf_score=float(pt.score),
                    dense_top_score=dense_top_score,
                )
            )

        logger.info(
            "qdrant_search",
            hits=len(results),
            filtered=filtered,
            top_k=top_k,
            dense_top_score=dense_top_score,
            collection=self._settings.qdrant_collection,
        )
        return results
