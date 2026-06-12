"""Hugging Face Inference API embeddings client.

Calls ``BAAI/bge-large-en-v1.5`` via the HF Inference feature-extraction
endpoint and returns 1024-d float vectors.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, get_settings
from core.errors import EmbeddingError, EmbeddingQuotaError
from core.logging_config import get_logger

logger = get_logger(__name__)


def _is_hf_retryable(exc: BaseException) -> bool:
    """Return True for transient HF errors. Quota exhaustion (402) is not retryable."""
    if isinstance(exc, EmbeddingQuotaError):
        return False
    return isinstance(exc, httpx.TransportError | httpx.HTTPStatusError | EmbeddingError)


class HuggingFaceEmbedder:
    """Thin async wrapper around the HF feature-extraction endpoint."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        token = self._settings.huggingface_api_token.get_secret_value()
        self._client = httpx.AsyncClient(
            base_url="https://router.huggingface.co/hf-inference",
            timeout=self._settings.irs_request_timeout_seconds,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "X-Wait-For-Model": "true",
            },
            transport=transport,
        )
        self._semaphore = asyncio.Semaphore(self._settings.hf_embed_concurrency)

    async def __aenter__(self) -> HuggingFaceEmbedder:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> tuple[float, ...]:
        """Return a single embedding vector."""

        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return embeddings for a batch. Each text is sent individually so a
        malformed entry cannot poison the rest of the batch."""

        results: list[tuple[float, ...]] = [() for _ in texts]

        async def _one(index: int, payload: str) -> None:
            async with self._semaphore:
                results[index] = await self._embed_one(payload)

        await asyncio.gather(*(_one(i, t) for i, t in enumerate(texts)))
        return results

    async def _embed_one(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        url = f"/models/{self._settings.embedding_model}"
        body = {
            "inputs": text,
            "options": {"wait_for_model": True, "use_cache": True},
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.irs_max_retries + 1),
            wait=wait_exponential(multiplier=2.0, min=2.0, max=60.0),
            retry=retry_if_exception(_is_hf_retryable),
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "hf_embed_retrying",
                attempt=rs.attempt_number,
                wait_seconds=round(rs.next_action.sleep, 1) if rs.next_action else None,
                error=str(rs.outcome.exception()) if rs.outcome else None,
            ),
        ):
            with attempt:
                response = await self._client.post(url, json=body)

                if response.status_code == 402:
                    logger.error(
                        "hf_embed_quota_exceeded",
                        status=402,
                        model=self._settings.embedding_model,
                        hint="Upgrade HF plan or wait for quota reset.",
                    )
                    raise EmbeddingQuotaError(
                        "HuggingFace embedding quota exceeded (HTTP 402). "
                        "Upgrade your HF plan or wait for the quota window to reset."
                    )

                if response.status_code == 429:
                    logger.warning(
                        "hf_embed_rate_limited",
                        status=429,
                        model=self._settings.embedding_model,
                    )
                    raise EmbeddingError("HF rate limited (HTTP 429); retrying with backoff")

                if response.status_code == 503:
                    raise EmbeddingError("HF inference endpoint cold; retrying")

                response.raise_for_status()
                vector = _coerce_vector(response.json())
                if len(vector) != self._settings.embedding_dimension:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch: got {len(vector)}, "
                        f"expected {self._settings.embedding_dimension}"
                    )
                return vector
        raise EmbeddingError("Embedding call exhausted retries")  # pragma: no cover


def _coerce_vector(payload: object) -> tuple[float, ...]:
    """Accept any of the response shapes HF returns and produce a flat tuple."""

    if not isinstance(payload, list) or not payload:
        raise EmbeddingError("Embedding payload must be a non-empty list")

    head = payload[0]
    if isinstance(head, int | float):
        return tuple(float(component) for component in payload)

    if isinstance(head, list):
        if head and isinstance(head[0], int | float):
            return tuple(float(component) for component in head)
        if head and isinstance(head[0], list) and isinstance(head[0][0], int | float):
            return tuple(float(component) for component in head[0])

    raise EmbeddingError("Embedding payload has unsupported shape")
