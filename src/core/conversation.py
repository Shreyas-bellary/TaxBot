"""Conversation turn helpers for multi-turn RAG.

The frontend owns chat history in localStorage. Each ``POST /v1/ask`` may
include recent prior turns so the query router and answer prompt can resolve
follow-ups. Nothing in this module is written to the database.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant"]

# Cap history so prompts stay bounded and injection surface stays limited.
MAX_HISTORY_TURNS = 12
MAX_TURN_CHARS = 2_000


class ChatTurn(BaseModel):
    """A single prior message from the client-owned chat transcript."""

    role: ChatRole
    content: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)


def normalize_history(turns: list[ChatTurn] | None) -> tuple[ChatTurn, ...]:
    """Trim, truncate, and cap prior turns for safe prompt injection."""

    if not turns:
        return ()

    cleaned: list[ChatTurn] = []
    for turn in turns[-MAX_HISTORY_TURNS:]:
        text = " ".join(turn.content.split()).strip()
        if not text:
            continue
        if len(text) > MAX_TURN_CHARS:
            text = text[: MAX_TURN_CHARS - 1].rstrip() + "…"
        cleaned.append(ChatTurn(role=turn.role, content=text))
    return tuple(cleaned)


def format_history_block(turns: tuple[ChatTurn, ...]) -> str:
    """Render prior turns for the answer-generation prompt."""

    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns:
        label = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


def format_router_user_message(
    current_fenced: str,
    turns: tuple[ChatTurn, ...],
) -> str:
    """Build the query-router user message for the current turn + history."""

    history_block = format_history_block(turns)
    if not history_block:
        return current_fenced
    return (
        "CONVERSATION HISTORY (prior turns — use for follow-up intent only)\n"
        "============================================================\n"
        f"{history_block}\n\n"
        "CURRENT QUESTION\n"
        "================\n"
        f"{current_fenced}"
    )
