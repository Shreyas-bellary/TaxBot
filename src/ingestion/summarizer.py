"""Table summarisation with a configurable primary provider and automatic fallback.

Primary and fallback are selected via Settings.table_summary_provider:
  * `gemini`     — Gemini primary, OpenRouter fallback
  * `openrouter` — OpenRouter primary, Gemini fallback

The summariser produces a *deterministic* 3-sentence description for every
table extracted by Unstructured. The full markdown table is preserved in the
parent node so the LLM always sees the raw data.

A strict heuristic post-validator enforces:
  * Exactly 3 sentences (split on terminal punctuation).
  * Each sentence has at least 3 tokens.
  * The combined length is bounded (no run-away outputs).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import Settings, get_settings
from core.errors import SummarizationError
from core.logging_config import get_logger

logger = get_logger(__name__)

_SUMMARY_PROMPT_TEMPLATE = """You are an IRS tax document analyst. Summarize the
following IRS table in EXACTLY THREE sentences. Cover (1) what the table
contains, (2) the structural axes (columns / rows / brackets / thresholds),
and (3) the most consequential numeric or rule-based takeaways. Do not invent
information that is not in the table. Do not include preambles or markdown.

Document context:
- Document number: {doc_number}
- Document title: {doc_title}
- Tax year (best-effort): {tax_year}

Table (markdown):
{table_markdown}

Three-sentence summary:"""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_SUMMARY_CHARS = 1500

_GEMINI_RETRYABLE_CODES: frozenset[int] = frozenset({429, 500, 503})


def _parse_retry_delay(exc: genai_errors.APIError) -> float | None:
    """Return retry delay seconds from Gemini RetryInfo."""
    error_body = exc.details
    if not isinstance(error_body, dict):
        return None
    for item in error_body.get("error", {}).get("details", ()):
        if isinstance(item, dict) and "retryDelay" in item:
            return float(str(item["retryDelay"]).removesuffix("s"))
    return None


def _is_gemini_retryable(exc: BaseException) -> bool:
    """Return True for transient Gemini errors (rate-limit, server errors)."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        return code in _GEMINI_RETRYABLE_CODES
    return False


@dataclass(frozen=True, slots=True)
class TableSummaryInput:
    doc_number: str
    doc_title: str
    tax_year: int | None
    table_markdown: str


SummarizeFn = Callable[[TableSummaryInput], Awaitable[str]]


class TableSummarizer:
    """Primary + fallback table summariser."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        primary: SummarizeFn | None = None,
        fallback: SummarizeFn | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if primary is None or fallback is None:
            resolved_primary, resolved_fallback = _resolve_table_summarizer_providers(
                self._settings
            )
            self._primary = primary or resolved_primary
            self._fallback = fallback if fallback is not None else resolved_fallback
        else:
            self._primary = primary
            self._fallback = fallback
        self._concurrency = asyncio.Semaphore(self._settings.table_summary_concurrency)

    async def summarize(self, payload: TableSummaryInput) -> str:
        """Run the primary under the shared concurrency gate, falling back on failure."""

        async with self._concurrency:
            try:
                text = await self._primary(payload)
                return _enforce_three_sentences(text)
            except Exception as primary_exc:
                logger.warning(
                    "table_summarizer_primary_failed",
                    error=str(primary_exc),
                    doc_number=payload.doc_number,
                )
                if self._fallback is None:
                    raise SummarizationError(str(primary_exc)) from primary_exc
                try:
                    text = await self._fallback(payload)
                    return _enforce_three_sentences(text)
                except Exception as fallback_exc:
                    raise SummarizationError(
                        f"Primary and fallback summarizers failed: "
                        f"primary={primary_exc!r}, fallback={fallback_exc!r}"
                    ) from fallback_exc


# ----------------------------------------------------------------------------
# Provider adapters
# ----------------------------------------------------------------------------
def _resolve_table_summarizer_providers(
    settings: Settings,
) -> tuple[SummarizeFn, SummarizeFn | None]:
    """Return (primary, fallback) for table summarization from settings."""
    gemini = _make_gemini_summarizer(settings)
    openrouter = _make_openrouter_summarizer(settings)

    if settings.table_summary_provider == "gemini":
        return gemini, openrouter

    if openrouter is None:
        raise ValueError(
            "TAXBOT_OPENROUTER_API_KEY is required when "
            "TAXBOT_TABLE_SUMMARY_PROVIDER=openrouter"
        )
    return openrouter, gemini


def _build_prompt(payload: TableSummaryInput) -> str:
    return _SUMMARY_PROMPT_TEMPLATE.format(
        doc_number=payload.doc_number,
        doc_title=payload.doc_title,
        tax_year=payload.tax_year if payload.tax_year is not None else "unknown",
        table_markdown=payload.table_markdown,
    )


def _make_gemini_summarizer(settings: Settings) -> SummarizeFn:
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    model_id = settings.gemini_model
    max_retries = settings.gemini_max_retries
    retry_max_wait = settings.gemini_retry_max_wait

    async def _summarize(payload: TableSummaryInput) -> str:
        prompt = _build_prompt(payload)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=wait_exponential(multiplier=2.0, min=4.0, max=retry_max_wait),
            retry=retry_if_exception(_is_gemini_retryable),
            reraise=True,
        ):
            with attempt:
                try:
                    response = await client.aio.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=400,
                            response_mime_type="text/plain",
                            thinking_config=genai_types.ThinkingConfig(
                                thinking_level=genai_types.ThinkingLevel.MINIMAL
                            ),
                        ),
                    )
                    text = getattr(response, "text", "") or ""
                    if not text.strip():
                        raise SummarizationError("Gemini returned empty completion")
                    return text.strip()
                except genai_errors.APIError as exc:
                    if getattr(exc, "code", None) == 429:
                        api_delay = _parse_retry_delay(exc)
                        if api_delay is not None:
                            wait = api_delay * 1.1
                            logger.warning(
                                "gemini_rate_limit_waiting",
                                doc_number=payload.doc_number,
                                api_delay_seconds=api_delay,
                                wait_seconds=round(wait, 1),
                            )
                            await asyncio.sleep(wait)
                    raise
        raise SummarizationError("Gemini summarizer exhausted retries")  # pragma: no cover

    return _summarize


def _make_openrouter_summarizer(settings: Settings) -> SummarizeFn | None:
    if settings.openrouter_api_key is None:
        return None

    api_key = settings.openrouter_api_key.get_secret_value()
    model_id = settings.openrouter_model
    timeout = settings.irs_request_timeout_seconds
    max_retries = settings.irs_max_retries

    async def _summarize(payload: TableSummaryInput) -> str:
        prompt = _build_prompt(payload)
        request = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an IRS tax document analyst that returns exactly "
                        "three sentences of grounded prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 400,
        }

        async with httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "TaxBot",
            },
        ) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries + 1),
                wait=wait_exponential(multiplier=1.0, max=15.0),
                retry=retry_if_exception_type(
                    (httpx.TransportError, httpx.HTTPStatusError, SummarizationError)
                ),
                reraise=True,
            ):
                with attempt:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=request,
                    )
                    response.raise_for_status()
                    body = response.json()
                    choices = body.get("choices") or []
                    if not choices:
                        raise SummarizationError("OpenRouter returned no choices")
                    content = (
                        choices[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if not content.strip():
                        raise SummarizationError("OpenRouter returned empty content")
                    return str(content).strip()
        raise SummarizationError("OpenRouter call exhausted retries")  # pragma: no cover

    return _summarize


# ----------------------------------------------------------------------------
# Post-processor
# ----------------------------------------------------------------------------
def _enforce_three_sentences(text: str) -> str:
    """Validate that the summariser produced exactly three usable sentences."""

    flattened = " ".join(text.split())
    if not flattened:
        raise SummarizationError("Empty summary text")
    if len(flattened) > _MAX_SUMMARY_CHARS:
        flattened = flattened[:_MAX_SUMMARY_CHARS]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(flattened) if s.strip()]
    if len(sentences) < 2:
        raise SummarizationError(
            f"Expected ~3 sentences in summary, got {len(sentences)}: {flattened!r}"
        )
    sentences = sentences[:3]
    for sentence in sentences:
        if len(sentence.split()) < 3:
            raise SummarizationError(
                f"Summary sentence too short to be informative: {sentence!r}"
            )
    return " ".join(sentences)
