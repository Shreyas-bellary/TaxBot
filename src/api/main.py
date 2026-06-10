"""FastAPI application exposing the TaxBot RAG pipeline.

The frontend will call ``POST /v1/ask`` with a free-form question and receive
either an answer + citations + parent context, or a typed 4xx error from one
of the security/retrieval guards.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core.config import Settings, get_settings
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
from core.models import GenerationResult, ParentNode, RetrievedContext
from core.repository import DocumentRepository
from core.retrieval import HybridRetriever
from core.security import InputGuard, OutputGuard
from ingestion.embeddings import HuggingFaceEmbedder

logger = get_logger(__name__)


class _AppState:
    settings: Settings
    database: Database
    embedder: HuggingFaceEmbedder
    repository: DocumentRepository
    retriever: HybridRetriever
    generator: AnswerGenerator


state = _AppState()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)
    state.settings = settings
    state.database = Database(settings)
    await state.database.connect()
    state.embedder = HuggingFaceEmbedder(settings)
    state.repository = DocumentRepository(state.database)
    state.retriever = HybridRetriever(state.repository, state.embedder, settings=settings)
    state.generator = AnswerGenerator(
        state.retriever,
        input_guard=InputGuard(settings),
        output_guard=OutputGuard(settings),
        settings=settings,
    )
    logger.info("api_ready")
    try:
        yield
    finally:
        await state.embedder.aclose()
        await state.database.close()


app = FastAPI(
    title="TaxBot",
    version="0.1.0",
    description="Deterministic IRS RAG API",
    lifespan=_lifespan,
)


class AskRequest(BaseModel):
    """Free-form user question."""

    query: str = Field(..., min_length=3, max_length=2000)


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


def _get_generator() -> AnswerGenerator:
    return state.generator


@app.get("/healthz", response_model=dict[str, str])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    generator: Annotated[AnswerGenerator, Depends(_get_generator)],
) -> AskResponse:
    try:
        result, context = await generator.answer_with_context(payload.query)
    except OutOfDomainQueryError as exc:
        return AskResponse(
            answer=str(exc),
            citations=[],
            used_parent_ids=[],
            parents=[],
            matched_child_ids=[],
        )
    except InjectionDetectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except OutputCitationError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except SecurityError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc

    return _to_response(result, context)


def _to_response(result: GenerationResult, context: RetrievedContext) -> AskResponse:
    used_ids = set(result.used_parent_ids)
    parents = [
        _parent_to_response(parent)
        for parent in context.parent_nodes
        if parent.id in used_ids
    ]

    return AskResponse(
        answer=result.answer,
        citations=[str(url) for url in result.citations],
        used_parent_ids=[str(pid) for pid in result.used_parent_ids],
        parents=parents,
        matched_child_ids=[str(cid) for cid in context.matched_child_ids],
    )


def _parent_to_response(parent: ParentNode) -> CitedParent:
    return CitedParent(
        parent_id=str(parent.id),
        text_content=parent.text_content,
        metadata=dict(parent.metadata),
    )
