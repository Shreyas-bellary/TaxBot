"""OWASP-aligned prompt-injection defences for TaxBot.

Two guards are implemented:

* :class:`InputGuard` runs on every raw user query *before* it touches
  embeddings, retrieval, or the LLM. It strips control characters, enforces
  a length budget, runs deterministic regex checks against well-known
  prompt-injection signatures, and wraps the cleaned text in cryptographic
  delimiters so the answer-generation prompt cannot accidentally interpret
  user content as system instructions.

* :class:`OutputGuard` evaluates LLM completions before they leave the
  service. It requires every answer to (a) cite at least one ``[Doc-N]``
  chunk ID that maps to a context parent, except the canonical
  :data:`NOT_FOUND_ANSWER` fallback, or (b) avoid leaking the user-query
  fences. Failures are converted into :class:`OutputCitationError` so
  callers can fail safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from pydantic import HttpUrl

from core.config import Settings, get_settings
from core.errors import InjectionDetectedError, OutputCitationError
from core.logging_config import get_logger
from core.models import GenerationResult, RetrievedContext

logger = get_logger(__name__)

# Matches [Doc-1], [Doc-12], etc.
_DOC_CITATION_RE = re.compile(r"\[Doc-(\d+)\]")
NOT_FOUND_ANSWER: Final[str] = (
    "I could not find an authoritative answer in the retrieved IRS documents."
)
MAX_QUERY_LENGTH: Final[int] = 2_000
MIN_QUERY_LENGTH: Final[int] = 10

# Curated prompt-injection signatures. The regexes are deliberately
# case-insensitive and word-boundary aware so polite variants ("Please
# ignore the previous instructions") still trip the guard.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:the\s+|all\s+)*(?:previous|prior|above|earlier|preceding)\s+instructions?\b",
        r"\bdisregard\s+(?:the\s+|all\s+)*(?:previous|prior|above|earlier|preceding)\s+instructions?\b",
        r"\b(?:ignore|disregard)\s+all\s+(?:previous|prior|above|earlier|preceding)\s+instructions?\b",
        r"\boverride\s+(?:the\s+)?(?:system|developer|safety)\b",
        r"\b(?:system|developer)\s+override\b",
        r"\b(?:you\s+are\s+now|act\s+as)\s+(?:in\s+)?(?:developer|dan|jailbroken|root|admin)\s+mode\b",
        r"\bdeveloper\s+mode\b",
        r"\bbypass(?:\s+(?:the\s+)?(?:safety|content|filter|guardrails?))?\b",
        r"\b(?:reveal|print|show|leak|expose|disclose)\s+(?:the\s+)?(?:hidden|secret|developer|system|internal)\b",
        r"\b(?:system|developer|hidden|secret)\s+prompt\b",
        r"\bjailbreak\b",
        r"\bDAN\s+(?:mode|prompt)\b",
        r"<\s*\|?\s*(?:system|sys|admin)\s*\|?\s*>",
        r"```\s*(?:system|developer|admin)\b",
    )
)

# Characters we strip outright. ``\x00``-``\x1f`` (except common whitespace)
# would otherwise allow homoglyph / control-character injection.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class SanitizedQuery:
    """The output of :class:`InputGuard`. Always safe to feed into prompts."""

    cleaned_text: str
    fenced_prompt_section: str
    start_tag: str
    end_tag: str


class InputGuard:
    """Validate and fence user queries before they reach retrieval/LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def sanitize(self, raw_query: str) -> SanitizedQuery:
        """Return a sanitised, fenced rendering of ``raw_query``."""

        if not isinstance(raw_query, str):
            raise InjectionDetectedError("Query must be a string")

        without_controls = _CONTROL_CHAR_RE.sub("", raw_query)
        flattened = _WHITESPACE_RE.sub(" ", without_controls).strip()

        if len(flattened) < MIN_QUERY_LENGTH:
            raise InjectionDetectedError("Query is too short to be meaningful")
        if len(flattened) > MAX_QUERY_LENGTH:
            raise InjectionDetectedError(
                f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
            )

        start = self._settings.user_query_start_tag
        end = self._settings.user_query_end_tag
        if start in flattened or end in flattened:
            raise InjectionDetectedError(
                "Query attempts to forge the user-query fence tags"
            )

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(flattened):
                logger.warning(
                    "prompt_injection_detected",
                    pattern=pattern.pattern,
                    query_preview=flattened[:120],
                )
                raise InjectionDetectedError(
                    "Query matches a known prompt-injection signature"
                )

        fenced = f"[{start}]\n{flattened}\n[{end}]"
        return SanitizedQuery(
            cleaned_text=flattened,
            fenced_prompt_section=fenced,
            start_tag=start,
            end_tag=end,
        )


class OutputGuard:
    """Validate LLM completions against the retrieved context."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def validate(
        self,
        *,
        answer: str,
        context: RetrievedContext,
    ) -> GenerationResult:
        """Return a :class:`GenerationResult` if the answer is safe to ship."""

        if not isinstance(answer, str) or not answer.strip():
            raise OutputCitationError("Empty LLM completion")

        if self._settings.user_query_start_tag in answer or self._settings.user_query_end_tag in answer:
            raise OutputCitationError("Completion leaked the user-query fence tags")

        if answer.strip() == NOT_FOUND_ANSWER:
            return GenerationResult(
                answer=NOT_FOUND_ANSWER,
                citations=(),
                used_parent_ids=(),
            )

        # Map [Doc-N] chunk IDs cited in the answer back to source URLs.
        cited_indices = {
            int(m.group(1))
            for m in _DOC_CITATION_RE.finditer(answer)
        }
        parents = context.parent_nodes
        cited_urls: list[HttpUrl] = []
        used_parents: list[UUID] = []
        seen_urls: set[str] = set()
        for idx in sorted(cited_indices):
            if idx < 1 or idx > len(parents):
                continue
            parent = parents[idx - 1]
            source_url = parent.metadata.get("source_url")
            if not isinstance(source_url, str) or not source_url:
                continue
            used_parents.append(parent.id)
            if source_url not in seen_urls:
                seen_urls.add(source_url)
                cited_urls.append(HttpUrl(source_url))

        if not cited_urls:
            logger.warning(
                "output_citation_missing",
                query_preview=context.query[:120],
                allowed_doc_ids=[f"Doc-{i}" for i in range(1, len(parents) + 1)],
            )
            raise OutputCitationError(
                "Completion did not cite any [Doc-N] chunk ID"
            )

        return GenerationResult(
            answer=answer.strip(),
            citations=tuple(cited_urls),
            used_parent_ids=tuple(used_parents),
        )
