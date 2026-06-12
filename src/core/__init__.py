"""Core domain primitives for TaxBot: configuration, models, persistence, retrieval, security."""

from core.config import Settings, get_settings
from core.errors import (
    EmbeddingError,
    EmbeddingQuotaError,
    IngestionError,
    InjectionDetectedError,
    OutOfDomainQueryError,
    OutputCitationError,
    RetrievalError,
    SecurityError,
    SummarizationError,
    TaxBotError,
    UnstructuredParseError,
    ValidationFailure,
)
from core.logging_config import configure_logging, get_logger

__all__ = [
    "EmbeddingError",
    "EmbeddingQuotaError",
    "IngestionError",
    "InjectionDetectedError",
    "OutOfDomainQueryError",
    "OutputCitationError",
    "RetrievalError",
    "SecurityError",
    "Settings",
    "SummarizationError",
    "TaxBotError",
    "UnstructuredParseError",
    "ValidationFailure",
    "configure_logging",
    "get_logger",
    "get_settings",
]
