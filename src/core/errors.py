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


class SummarizationError(IngestionError):
    """Raised when both the primary and fallback table summarizers fail."""


class EmbeddingError(IngestionError):
    """Raised when an embedding call returns a malformed or empty vector."""


class RetrievalError(TaxBotError):
    """Raised when the hybrid retrieval pipeline cannot produce a context payload."""


class SecurityError(TaxBotError):
    """Base class for OWASP-aligned guardrail violations."""


class InjectionDetectedError(SecurityError):
    """Raised when a user input matches a known prompt-injection signature."""


class OutputCitationError(SecurityError):
    """Raised when an LLM output fails the provenance-citation alignment check."""
