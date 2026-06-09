"""Shared Pydantic v2 models that define the typed contracts between
ingestion, retrieval, security, and evaluation layers.

These models are the canonical wire format inside the system. External I/O
(REST JSON, SQL rows, Unstructured payloads) is converted to/from these
classes at the boundary, after which downstream code can rely on strict
types and validators.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Matches IRS document numbers such as "Form 1040", "Form 1040 (Schedule A)",
# "Publication 17", "Instructions for Form 941", etc. We keep the regex
# permissive on purpose but require a leading category keyword.
_DOC_NUMBER_RE = re.compile(
    r"^(?:Form|Publication|Notice|Instructions?(?:\s+for(?:\s+Forms?)?)?|Schedule)\b",
    re.IGNORECASE,
)


class DocCategory(StrEnum):
    """Top-level taxonomy used to route documents and bias retrieval filters."""

    FORM = "form"
    INSTRUCTION = "instruction"
    PUBLICATION = "publication"
    NOTICE = "notice"
    OTHER = "other"


class NodeKind(StrEnum):
    """Discriminator for parent/child node payloads."""

    PARENT = "parent"
    CHILD = "child"


class _FrozenModel(BaseModel):
    """Base model used by stable wire types. Frozen to prevent mutation."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")


class IRSDocumentMetadata(_FrozenModel):
    """One row produced by the IRS Drupal Views AJAX endpoint.

    This is the strict contract the scraper is allowed to emit. Any row that
    fails validation must be rejected and logged rather than silently coerced.
    """

    doc_number: NonEmptyStr
    doc_title: NonEmptyStr
    revision_date: NonEmptyStr
    posted_date: NonEmptyStr
    pdf_url: HttpUrl

    @field_validator("doc_number")
    @classmethod
    def _validate_doc_number_prefix(cls, value: str) -> str:
        if not _DOC_NUMBER_RE.match(value):
            raise ValueError(
                f"doc_number {value!r} does not look like an IRS Forms/Publications identifier"
            )
        return value

    @property
    def category(self) -> DocCategory:
        """Derive the taxonomy category from the document number prefix."""

        lowered = self.doc_number.lower()
        if lowered.startswith("publication"):
            return DocCategory.PUBLICATION
        if lowered.startswith("notice"):
            return DocCategory.NOTICE
        if lowered.startswith("instruction"):
            return DocCategory.INSTRUCTION
        if lowered.startswith(("form", "schedule")):
            return DocCategory.FORM
        return DocCategory.OTHER

    @property
    def tax_year(self) -> int | None:
        """Best-effort tax year inference from revision_date.

        Revision dates on the IRS listing appear in formats like ``"2024"``,
        ``"Sep 2017"``, ``"Apr 2021"``. We extract the trailing four-digit year
        when present and return :data:`None` otherwise. Downstream code must
        treat ``None`` as "year unknown" and route accordingly.
        """

        match = re.search(r"(19|20)\d{2}", self.revision_date)
        if match is None:
            return None
        return int(match.group(0))


class IRSDocumentRecord(_FrozenModel):
    """Augmented record that pairs a scraper row with derived ingestion state."""

    metadata: IRSDocumentMetadata
    doc_id: UUID = Field(default_factory=uuid4)
    pdf_sha256: NonEmptyStr | None = None
    fetched_at: datetime | None = None

    @property
    def source_url(self) -> str:
        return str(self.metadata.pdf_url)


class ParentNode(_FrozenModel):
    """A large, highly contextual block stored in ``parent_nodes``.

    ``text_content`` holds whole sections or full markdown tables. The
    ``metadata`` JSON blob is denormalised on purpose so the FTS prefilter
    can scope by ``tax_year`` / ``form_number`` without joining child rows.
    """

    id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    text_content: NonEmptyStr
    metadata: dict[str, object]
    created_at: datetime | None = None


class ChildNode(_FrozenModel):
    """A small semantic unit (sentence / table summary) linked to a parent."""

    id: UUID = Field(default_factory=uuid4)
    parent_id: UUID
    text_summary: NonEmptyStr
    embedding: tuple[float, ...] = Field(default_factory=tuple)
    metadata: dict[str, object]
    created_at: datetime | None = None

    @field_validator("embedding")
    @classmethod
    def _embedding_is_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for component in value:
            if not isinstance(component, int | float) or component != component:  # NaN check
                raise ValueError("embedding contains a non-finite or non-numeric component")
        return value

    @model_validator(mode="after")
    def _embedding_dimension(self) -> ChildNode:
        if len(self.embedding) not in (0, 1024):
            raise ValueError(
                f"embedding length must be 0 (unset) or 1024, got {len(self.embedding)}"
            )
        return self


class RetrievedContext(_FrozenModel):
    """A fully assembled retrieval payload returned to the generation stage."""

    query: NonEmptyStr
    parent_nodes: tuple[ParentNode, ...]
    matched_child_ids: tuple[UUID, ...]
    source_urls: tuple[HttpUrl, ...]


class GenerationResult(_FrozenModel):
    """Validated LLM completion paired with the citation proofs that satisfy the
    output security gate."""

    answer: NonEmptyStr
    citations: tuple[HttpUrl, ...]
    used_parent_ids: tuple[UUID, ...]
