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
    supabase_url: AnyHttpUrl = Field(..., description="Supabase project URL.")
    supabase_service_role_key: SecretStr = Field(
        ...,
        description="Service-role key used for privileged Supabase REST calls.",
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
        description="Dimensionality of the embedding vectors written into pgvector.",
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
        description="Optional OpenRouter fallback for table summarization.",
    )
    openrouter_model: str = Field(
        default="google/gemini-flash-1.5",
        description="OpenRouter model id used when Gemini Flash is unavailable.",
    )

    answer_llm_provider: AnswerProvider = Field(
        default="gemini",
        description="Provider used for final answer synthesis.",
    )
    answer_llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Final answer model identifier.",
    )

    backfill_oldest_tax_year: int = Field(
        default=2020,
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
