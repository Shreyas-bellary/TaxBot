"""Strict-typed runtime configuration for TaxBot.

Configuration is loaded via `pydantic-settings` from environment variables
prefixed with ``TAXBOT_``. The settings object is immutable so that ingestion
workers and the retrieval layer cannot mutate global state at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AnswerProvider = Literal["gemini", "openrouter"]


class Settings(BaseSettings):
    """Centralised, immutable runtime configuration.

    The settings model deliberately fails closed: every secret is typed as
    :class:`SecretStr` and every URL is validated. Missing required values
    raise at startup rather than producing silent fallbacks at request time.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TAXBOT_",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    postgres_dsn: PostgresDsn = Field(
        ...,
        description="Async Postgres DSN used by asyncpg for parent/child writes.",
    )

    unstructured_api_key: SecretStr = Field(..., description="Hosted Unstructured API key.")
    unstructured_api_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://api.unstructuredapp.io"),
        description="Hosted Unstructured API base URL.",
    )

    huggingface_api_token: SecretStr = Field(
        ...,
        description="HF Inference API token used for embeddings.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace embedding model identifier.",
    )
    embedding_dimension: int = Field(
        default=1024,
        ge=1,
        description="Dimensionality of the dense embedding vectors (must match Qdrant collection).",
    )

    gemini_api_key: SecretStr = Field(
        ...,
        description="Google AI Studio API key for Gemini Flash table summaries.",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini Flash model id used for table summaries and answer synthesis.",
    )

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "OpenRouter API key. Required when table_summary_provider or "
            "answer_llm_provider is openrouter; otherwise used as table-summary fallback."
        ),
    )
    openrouter_model: str = Field(
        default="google/gemini-flash-1.5",
        description="OpenRouter model id for table summarization and answer synthesis.",
    )

    table_summary_provider: AnswerProvider = Field(
        default="gemini",
        description=(
            "Primary LLM provider for table summarization during ingest. "
            "The other provider is always used as fallback when configured."
        ),
    )

    answer_llm_provider: AnswerProvider = Field(
        default="gemini",
        description="Provider used for final answer synthesis.",
    )
    answer_llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Final answer model identifier.",
    )

    router_llm_provider: AnswerProvider = Field(
        default="gemini",
        description=(
            "LLM provider for the query router (domain gate + filter extraction). "
            "Use a cheap/fast model; the router call precedes every retrieval."
        ),
    )
    router_llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Model id for the query router LLM.",
    )

    eval_judge_provider: AnswerProvider = Field(
        default="gemini",
        description="Provider used by the Ragas evaluation judge LLM.",
    )
    eval_judge_model: str = Field(
        default="gemini-2.0-flash",
        description="Model identifier for the Ragas evaluation judge LLM.",
    )

    backfill_oldest_tax_year: int = Field(
        default=2023,
        ge=1990,
        le=2100,
        description="Inclusive lower bound for tax_year during historical backfill.",
    )
    language_allowlist: str = Field(
        default="en",
        description="Comma-separated language tags to keep; '*' disables filtering.",
    )
    narrative_content_filter_enabled: bool = Field(
        default=True,
        description=(
            "Enable Layer 1 deterministic narrative hygiene: drop IRS print/proof "
            "metadata and trailing index sections before parent/child chunking."
        ),
    )
    publication_max_pages: int = Field(
        default=200,
        ge=0,
        description=(
            "Skip publications with more than this many PDF pages during ingest. "
            "Set to 0 to disable and process all publications regardless of length."
        ),
    )

    irs_request_throttle_seconds: float = Field(
        default=1.5,
        ge=0.0,
        le=30.0,
        description="Per-page sleep between IRS AJAX requests.",
    )
    irs_request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Per-request HTTP timeout for IRS AJAX and PDF downloads.",
    )
    irs_max_retries: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Bounded retry budget for transient IRS HTTP failures.",
    )

    # --- Rate-limit resilience ---
    gemini_max_retries: int = Field(
        default=6,
        ge=0,
        le=20,
        description=(
            "Maximum retry attempts for Gemini API calls on 429/500/503 responses. "
            "Applies to both table summarization and answer generation."
        ),
    )
    gemini_retry_max_wait: float = Field(
        default=60.0,
        gt=0.0,
        description=(
            "Maximum seconds to wait between Gemini retry attempts (exponential backoff ceiling). "
            "Increase when hitting sustained 429 quota limits during backfill."
        ),
    )
    hf_embed_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description=(
            "Maximum simultaneous HuggingFace embedding requests across all in-flight documents. "
            "Reduce to 1-2 when hitting HF free-tier rate limits during backfill."
        ),
    )
    table_summary_concurrency: int = Field(
        default=1,
        ge=1,
        le=32,
        description=(
            "Maximum simultaneous Gemini table-summary calls across all in-flight documents. "
        ),
    )
    qdrant_upsert_batch_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description=(
            "Maximum number of Qdrant points sent in a single upsert request. "
            "Large publications can produce 500+ child nodes; batching avoids timeouts."
        ),
    )
    qdrant_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description=(
            "HTTP timeout (seconds) for Qdrant Cloud API calls. "
            "Increase for large batched upserts over slow links."
        ),
    )
    qdrant_upsert_max_retries: int = Field(
        default=4,
        ge=0,
        le=10,
        description=(
            "Retry budget for transient Qdrant upsert failures (write/read timeouts, "
            "transport errors)."
        ),
    )

    cors_allow_origins: str = Field(
        default="http://localhost:5173",
        description="Comma-separated browser origins allowed to call the API.",
    )

    # --- Rate limiting (Postgres-backed, per-IP) ---
    rate_limit_enabled: bool = Field(
        default=True,
        description=(
            "Enable the Postgres-backed per-IP daily answer quota on /v1/ask. "
            "Quota is shared across all instances via private.ip_daily_rate_limits."
        ),
    )
    rate_limit_answers_per_day: int = Field(
        default=3,
        ge=1,
        le=10_000,
        description="Number of successful answers allowed per client IP per UTC day.",
    )
    rate_limit_trust_forwarded_for: bool = Field(
        default=False,
        description=(
            "Trust the X-Forwarded-For header for client IP extraction. "
            "On Cloud Run (direct ingress) set to True: GFE appends the real "
            "client IP as the LAST entry, which is the only trusted value. "
            "Leave False when the API is directly reachable without a trusted proxy."
        ),
    )

    # --- Conversation History ---
    conversation_history_max_turns: int = Field(
        default=8,
        ge=0,
        le=50,
        description=(
            "Maximum number of prior chat messages (user+assistant) accepted from the "
            "client and used as transient context. History is never persisted server-side."
        ),
    )
    conversation_history_max_chars: int = Field(
        default=800,
        ge=100,
        le=8_000,
        description="Per-message character cap applied to client-supplied history turns.",
    )
    conversation_condense_enabled: bool = Field(
        default=True,
        description=(
            "Rewrite follow-up questions into standalone queries (using prior turns) "
            "before retrieval, improving recall on context-dependent follow-ups."
        ),
    )

    user_query_start_tag: str = Field(
        default="USER_QUERY_START_5f3c1e",
        min_length=8,
        description="Opening fence used to delimit untrusted user input inside prompts.",
    )
    user_query_end_tag: str = Field(
        default="USER_QUERY_END_5f3c1e",
        min_length=8,
        description="Closing fence used to delimit untrusted user input inside prompts.",
    )

    # --- Qdrant vector store ---
    qdrant_url: AnyHttpUrl = Field(
        ...,
        description="Qdrant Cloud (or local) cluster URL, e.g. https://xxx.qdrant.io:6333.",
    )
    qdrant_api_key: SecretStr = Field(
        ...,
        description="Qdrant API key for the cluster.",
    )
    qdrant_collection: str = Field(
        default="taxbot_child_nodes",
        description="Qdrant collection that holds dense+sparse child-node vectors.",
    )
    bm25_model: str = Field(
        default="Qdrant/bm25",
        description="fastembed sparse model used for BM25 keyword retrieval.",
    )
    retrieval_top_k_children: int = Field(
        default=12,
        ge=1,
        le=200,
        description="Number of candidate child nodes fetched from Qdrant per retrieval pass.",
    )
    retrieval_top_k_parents: int = Field(
        default=6,
        ge=1,
        le=50,
        description=(
            "Maximum unique parent nodes assembled into RetrievedContext"
        ),
    )
    retrieval_rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant k used in Qdrant fusion. Higher k flattens rank differences.",
    )

    retrieval_confidence_gate_enabled: bool = Field(
        default=True,
        description="Enable Layer 2 retrieval-confidence rejection for weak/off-topic matches.",
    )
    retrieval_min_hybrid_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum hybrid_score required for the top-ranked retrieval hit.",
    )
    retrieval_min_score_gap: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum top1-top2 hybrid_score gap; rejects ambiguous matches.",
    )

    # --- Reranker ---
    reranker_enabled: bool = Field(
        default=False,
        description=(
            "Enable the cross-encoder rerank step after Qdrant RRF and before parent expansion. "
            "Requires reranker_model_path to point to the fine-tuned LoRA adapter directory."
        ),
    )
    reranker_model_path: str = Field(
        default="scripts/finetuned_model",
        description=(
            "Path to the fine-tuned LoRA adapter directory "
            "(must contain adapter_config.json and adapter_model.safetensors)."
        ),
    )
    reranker_top_k: int = Field(
        default=12,
        ge=1,
        le=200,
        description="Number of child nodes to keep after reranking (before parent expansion).",
    )

    faithfulness_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    answer_relevancy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    context_precision_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    context_recall_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)

    @field_validator("language_allowlist")
    @classmethod
    def _normalise_language_allowlist(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def cors_origins(self) -> tuple[str, ...]:
        """Parsed list of allowed CORS origins."""

        return tuple(
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        )

    @property
    def language_tags(self) -> frozenset[str]:
        """Parsed allowlist of permitted language tags."""

        if self.language_allowlist == "*":
            return frozenset()
        return frozenset(
            tag.strip() for tag in self.language_allowlist.split(",") if tag.strip()
        )

    @property
    def multilingual_enabled(self) -> bool:
        return self.language_allowlist == "*" or len(self.language_tags) > 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated settings instance."""

    return Settings()
