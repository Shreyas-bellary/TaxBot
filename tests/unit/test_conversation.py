"""Unit tests for ephemeral conversation helpers (no DB)."""

from __future__ import annotations

from core.conversation import (
    ChatTurn,
    format_history_block,
    format_router_user_message,
    normalize_history,
)


def test_normalize_history_caps_and_trims() -> None:
    turns = [
        ChatTurn(role="user", content=f"  Question {i}  ")
        for i in range(20)
    ]
    result = normalize_history(turns)
    assert len(result) == 12
    assert result[0].content == "Question 8"
    assert result[-1].content == "Question 19"


def test_format_history_block() -> None:
    block = format_history_block(
        (
            ChatTurn(role="user", content="Hello tax question"),
            ChatTurn(role="assistant", content="Here is an answer"),
        )
    )
    assert "User: Hello tax question" in block
    assert "Assistant: Here is an answer" in block


def test_format_router_user_message_includes_history() -> None:
    history = (
        ChatTurn(
            role="user",
            content="What is the standard deduction for tax year 2024?",
        ),
        ChatTurn(
            role="assistant",
            content="The standard deduction for tax year 2024 is $29,200.",
        ),
    )
    message = format_router_user_message(
        "[START]\nwhat about for 2025?\n[END]",
        history,
    )
    assert "CONVERSATION HISTORY" in message
    assert "standard deduction" in message.lower()
    assert "what about for 2025?" in message
