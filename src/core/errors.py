"""Typed exception hierarchy for TaxBot.

A single inheritance tree makes failure modes explicit so callers can
distinguish ingestion failures, retrieval issues, and security violations
without inspecting message strings.
"""

from __future__ import annotations


class TaxBotError(Exception):
    """Base class for all TaxBot domain errors."""


class ValidationFailure(TaxBotError):
    """Raised when a Pydantic boundary contract rejects external data."""


class IngestionError(TaxBotError):
    """Raised when document ingestion fails irrecoverably for a given record."""


class UnstructuredParseError(IngestionError):
    """Raised when the Unstructured Hosted API returns an unusable payload."""


class UnsupportedPDFError(IngestionError):
    """Raised when a downloaded PDF is not renderable content.

    Some IRS PDFs are XFA/AcroForms that require Adobe Reader 8+ and contain
    only a stub page with an error message instead of the actual document body.
    These cannot be parsed and should be silently skipped.
    """


class SummarizationError(IngestionError):
    """Raised when both the primary and fallback table summarizers fail."""


class EmbeddingError(IngestionError):
    """Raised when an embedding call returns a malformed or empty vector."""


class EmbeddingQuotaError(EmbeddingError):
    """Raised when the embedding provider reports quota exhaustion (HTTP 402).
    """

class RetrievalError(TaxBotError):
    """Raised when the hybrid retrieval pipeline cannot produce a context payload."""


class SecurityError(TaxBotError):
    """Base class for OWASP-aligned guardrail violations."""


class InjectionDetectedError(SecurityError):
    """Raised when a user input matches a known prompt-injection signature."""


class OutOfDomainQueryError(SecurityError):
    """Raised when a user query is outside TaxBot's tax-only scope."""


OUT_OF_DOMAIN_MESSAGE = (
    "I can only answer U.S. tax-related questions. "
    "Please ask about IRS forms, filings, deductions, credits, tax years, or related topics."
)


class OutputCitationError(SecurityError):
    """Raised when an LLM output fails the provenance-citation alignment check."""
