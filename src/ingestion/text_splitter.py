"""Deterministic text splitting helpers used by the table pipeline.

We build *parent* blocks by grouping Unstructured narrative elements into
~1500-character sections, and we build *child* sentences by splitting on
sentence boundaries with a hard upper bound on length.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from core.logging_config import get_logger
from ingestion.unstructured_parser import NarrativeBlock

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

DEFAULT_PARENT_TARGET_CHARS = 1500
DEFAULT_PARENT_MAX_CHARS = 2200
DEFAULT_CHILD_TARGET_CHARS = 280
DEFAULT_CHILD_MAX_CHARS = 480


def group_narratives_into_parents(
    narratives: Iterable[NarrativeBlock],
    *,
    target_chars: int = DEFAULT_PARENT_TARGET_CHARS,
    max_chars: int = DEFAULT_PARENT_MAX_CHARS,
) -> list[str]:
    """Concatenate narrative blocks into ~target_chars sized parent texts."""

    buffer: list[str] = []
    current = ""
    for block in narratives:
        candidate = (current + "\n\n" + block.text).strip() if current else block.text
        if len(candidate) >= target_chars:
            if len(candidate) <= max_chars or not current:
                buffer.append(candidate)
                current = ""
            else:
                buffer.append(current)
                current = block.text
        else:
            current = candidate

    if current:
        buffer.append(current)
    return buffer


def split_into_child_sentences(
    text: str,
    *,
    target_chars: int = DEFAULT_CHILD_TARGET_CHARS,
    max_chars: int = DEFAULT_CHILD_MAX_CHARS,
) -> list[str]:
    """Split a parent block into small, embed-friendly sentence groups."""

    flattened = " ".join(text.split())
    if not flattened:
        return []
    raw_sentences = _SENTENCE_SPLIT_RE.split(flattened)
    chunks: list[str] = []
    current = ""
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) >= target_chars:
            if len(candidate) <= max_chars or not current:
                chunks.append(candidate)
                current = ""
            else:
                chunks.append(current)
                current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
