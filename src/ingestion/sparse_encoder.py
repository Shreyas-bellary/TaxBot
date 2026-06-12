"""BM25 sparse-vector encoder backed by fastembed.

:class:`SparseEncoder` wraps ``fastembed.SparseTextEmbedding`` and exposes
async helpers that run the CPU-bound encoding in a thread pool so it is
compatible with the rest of the async ingestion + retrieval pipeline.

Two call paths:
- :meth:`embed_documents` — ingest time; uses the standard ``embed`` method
  which applies IDF weighting on the corpus being indexed.
- :meth:`embed_query` — query time; uses ``query_embed`` which applies the
  query-side token weights.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from fastembed import SparseTextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client.models import SparseVector

from core.config import Settings, get_settings
from core.logging_config import get_logger

logger = get_logger(__name__)


def _to_sparse_vector(emb: SparseEmbedding) -> SparseVector:
    return SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )


class SparseEncoder:
    """Thread-safe, async-compatible BM25 encoder via fastembed."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: SparseTextEmbedding | None = None

    def _get_model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(model_name=self._settings.bm25_model)
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Encode a list of document texts for indexing (BM25 document weights)."""

        if not texts:
            return []

        def _encode() -> list[SparseVector]:
            model = self._get_model()
            results = list(model.embed(list(texts)))
            return [_to_sparse_vector(e) for e in results]

        vectors = await asyncio.to_thread(_encode)
        logger.info(
            "sparse_encode",
            count=len(texts),
            model=self._settings.bm25_model,
            mode="document",
        )
        return vectors

    async def embed_query(self, text: str) -> SparseVector:
        """Encode a single query string (BM25 query weights)."""

        def _encode() -> SparseVector:
            model = self._get_model()
            results = list(model.query_embed(text))
            return _to_sparse_vector(results[0])

        vector = await asyncio.to_thread(_encode)
        logger.debug(
            "sparse_encode",
            count=1,
            model=self._settings.bm25_model,
            mode="query",
        )
        return vector
