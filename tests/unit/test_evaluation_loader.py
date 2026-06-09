"""Tests for the Ragas fixture loader."""

from __future__ import annotations

from evaluation.validate import load_cases


def test_load_cases_returns_at_least_20() -> None:
    cases = load_cases()
    assert len(cases) >= 20
    ids = {c.id for c in cases}
    assert len(ids) == len(cases), "case ids must be unique"
    for case in cases:
        assert case.question.strip()
        assert case.ground_truth.strip()
