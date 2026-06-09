"""Structured logging setup.

We standardise on :mod:`structlog` so that every emitted log line is a JSON
object with keyed metadata. This makes ingestion runs, retrieval traces, and
security guard decisions trivially queryable in Loki / Datadog / etc.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def _drop_color_codes(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Strip ANSI escapes which can poison JSON logs in CI environments."""

    for key, value in list(event_dict.items()):
        if isinstance(value, str) and "\x1b[" in value:
            event_dict[key] = value.encode("ascii", "ignore").decode("ascii")
    return event_dict


def configure_logging(*, level: str = "INFO", as_json: bool = True) -> None:
    """Configure stdlib + structlog with a single shared pipeline.

    Calling this function more than once is idempotent.
    """

    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _drop_color_codes,
    ]

    if as_json:
        renderer: Processor = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured, bound structlog logger."""

    return structlog.get_logger(name)
