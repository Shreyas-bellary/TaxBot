"""FastAPI application exposing the TaxBot RAG pipeline.

The frontend calls ``POST /v1/ask`` with a free-form question and optional
conversation history (client-owned; never persisted server-side).
Responses include the answer, citations, parent context, and remaining
free-answer quota for the caller IP.
"""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import Settings, get_settings
from core.conversation import MAX_HISTORY_TURNS, ChatTurn
from core.db import Database
from core.errors import (
    InjectionDetectedError,
    OutOfDomainQueryError,
    OutputCitationError,
    RetrievalError,
    SecurityError,
)
from core.generation import AnswerGenerator
from core.logging_config import configure_logging, get_logger
from core.models import GenerationResult, RetrievedContext
from core.rate_limit import IpDailyRateLimiter, RateLimitDecision
from core.repository import DocumentRepository
from core.reranker import ChildReranker
from core.retrieval import HybridRetriever
from core.security import InputGuard, OutputGuard
from core.vector_store import QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder
from ingestion.sparse_encoder import SparseEncoder

logger = get_logger(__name__)


class _AppState:
    settings: Settings
    database: Database
    embedder: HuggingFaceEmbedder
    vector_store: QdrantVectorStore
    sparse_encoder: SparseEncoder
    repository: DocumentRepository
    reranker: ChildReranker | None
    retriever: HybridRetriever
    generator: AnswerGenerator
    rate_limiter: IpDailyRateLimiter


state = _AppState()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)
    state.settings = settings
    state.database = Database(settings)
    await state.database.connect()
    state.embedder = HuggingFaceEmbedder(settings)
    state.vector_store = QdrantVectorStore(settings)
    await state.vector_store.ensure_collection()
    state.sparse_encoder = SparseEncoder(settings)
    state.repository = DocumentRepository(state.database)
    state.reranker = (
        ChildReranker(
            settings.reranker_model_path,
            hf_token=settings.huggingface_api_token.get_secret_value(),
        )
        if settings.reranker_enabled
        else None
    )
    state.retriever = HybridRetriever(
        state.repository,
        state.embedder,
        state.vector_store,
        state.sparse_encoder,
        reranker=state.reranker,
        settings=settings,
    )
    state.generator = AnswerGenerator(
        state.retriever,
        input_guard=InputGuard(settings),
        output_guard=OutputGuard(settings),
        settings=settings,
    )
    state.rate_limiter = IpDailyRateLimiter(
        limit=settings.rate_limit_answers_per_day,
        database=state.database,
    )
    logger.info(
        "api_ready",
        rate_limit_enabled=settings.rate_limit_enabled,
        rate_limit_per_day=settings.rate_limit_answers_per_day,
    )
    try:
        yield
    finally:
        await state.embedder.aclose()
        await state.vector_store.aclose()
        await state.database.close()


app = FastAPI(
    title="TaxBot",
    version="0.1.0",
    description="Deterministic IRS RAG API",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
)


class AskRequest(BaseModel):
    """Free-form user question with optional client-owned chat history."""

    query: str = Field(..., min_length=3, max_length=2000)
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description=(
            "Prior turns from the current chat session."
        ),
    )


class CitedParent(BaseModel):
    parent_id: str
    text_content: str
    metadata: dict[str, object]


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    used_parent_ids: list[str]
    parents: list[CitedParent]
    matched_child_ids: list[str]
    rate_limit: dict[str, int | str] | None = None


def _get_generator() -> AnswerGenerator:
    return state.generator


def _parse_ip(raw: str) -> str | None:
    """Return the normalised IP string if ``raw`` is a valid IPv4/IPv6 address, else None."""
    try:
        return str(ipaddress.ip_address(raw.strip()))
    except ValueError:
        return None


def _client_ip(request: Request) -> str:
    """Extract a normalised client IP from the request.
    With direct connections (no trusted proxy) we use ``request.client.host``
    instead to avoid header-spoofing attacks.
    """
    if state.settings.rate_limit_trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        for candidate in reversed(forwarded.split(",")):
            ip = _parse_ip(candidate)
            if ip is not None:
                return ip

    if request.client and request.client.host:
        ip = _parse_ip(request.client.host)
        if ip is not None:
            return ip

    return "unknown"


def _apply_rate_headers(response: Response, decision: RateLimitDecision) -> None:
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    response.headers["X-RateLimit-Reset"] = decision.reset_at.isoformat()
    if not decision.allowed and decision.retry_after_seconds > 0:
        response.headers["Retry-After"] = str(decision.retry_after_seconds)


def _rate_limit_payload(decision: RateLimitDecision) -> dict[str, int | str]:
    return {
        "limit": decision.limit,
        "remaining": decision.remaining,
        "reset_at": decision.reset_at.isoformat(),
    }


@app.get("/healthz", response_model=dict[str, str])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/rate-limit", response_model=dict[str, int | str | bool])
async def rate_limit_status(request: Request) -> dict[str, int | str | bool]:
    """Return the caller's remaining free answers for today (no consume)."""

    if not state.settings.rate_limit_enabled:
        return {
            "enabled": False,
            "limit": state.rate_limiter.limit,
            "remaining": state.rate_limiter.limit,
            "reset_at": "",
        }
    decision = await state.rate_limiter.check(_client_ip(request))
    return {
        "enabled": True,
        "limit": decision.limit,
        "remaining": decision.remaining,
        "reset_at": decision.reset_at.isoformat(),
    }


@app.post("/v1/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    response: Response,
    generator: Annotated[AnswerGenerator, Depends(_get_generator)],
) -> AskResponse:
    client_ip = _client_ip(request)
    rate_decision: RateLimitDecision | None = None
    reserved = False

    if state.settings.rate_limit_enabled:
        rate_decision = await state.rate_limiter.consume(client_ip)
        _apply_rate_headers(response, rate_decision)
        if not rate_decision.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Daily free limit of {rate_decision.limit} answers reached. "
                    f"Try again after {rate_decision.reset_at.isoformat()} (UTC)."
                ),
                headers={
                    "X-RateLimit-Limit": str(rate_decision.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": rate_decision.reset_at.isoformat(),
                    "Retry-After": str(rate_decision.retry_after_seconds),
                },
            )
        reserved = True

    try:
        result, context = await generator.answer_with_context(
            payload.query,
            history=payload.history or None,
        )
    except OutOfDomainQueryError as exc:
        return AskResponse(
            answer=str(exc),
            citations=[],
            used_parent_ids=[],
            parents=[],
            matched_child_ids=[],
            rate_limit=_rate_limit_payload(rate_decision) if rate_decision else None,
        )
    except InjectionDetectedError as exc:
        if reserved:
            await state.rate_limiter.refund(client_ip)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except OutputCitationError as exc:
        if reserved:
            await state.rate_limiter.refund(client_ip)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except RetrievalError as exc:
        if reserved:
            await state.rate_limiter.refund(client_ip)
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except SecurityError as exc:
        if reserved:
            await state.rate_limiter.refund(client_ip)
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc

    return _to_response(result, context, rate_decision)


def _to_response(
    result: GenerationResult,
    context: RetrievedContext,
    rate_decision: RateLimitDecision | None,
) -> AskResponse:
    used_ids = set(result.used_parent_ids)
    parents: list[CitedParent] = []
    for index, parent in enumerate(context.parent_nodes, start=1):
        if parent.id not in used_ids:
            continue
        metadata = dict(parent.metadata)
        metadata["doc_index"] = index
        parents.append(
            CitedParent(
                parent_id=str(parent.id),
                text_content=parent.text_content,
                metadata=metadata,
            )
        )

    return AskResponse(
        answer=result.answer,
        citations=[str(url) for url in result.citations],
        used_parent_ids=[str(pid) for pid in result.used_parent_ids],
        parents=parents,
        matched_child_ids=[str(cid) for cid in context.matched_child_ids],
        rate_limit=_rate_limit_payload(rate_decision) if rate_decision else None,
    )
