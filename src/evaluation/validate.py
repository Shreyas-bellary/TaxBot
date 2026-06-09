"""Ragas evaluation gate.

Runs the 20-case TaxBot benchmark, computes Faithfulness, Answer Relevance,
Context Precision, and Context Recall, and asserts each metric exceeds the
threshold configured in :class:`Settings`. If any metric falls below the
gate, the process exits with a non-zero status so CI fails immediately.

The evaluator runs the live TaxBot pipeline end-to-end and uses Gemini
Flash as the Ragas judge LLM. Network and database calls are wrapped in
asyncio.gather so all cases evaluate concurrently up to ``max_concurrency``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from core.config import Settings, get_settings
from core.db import Database
from core.generation import AnswerGenerator
from core.logging_config import configure_logging, get_logger
from core.repository import DocumentRepository
from core.retrieval import HybridRetriever
from ingestion.embeddings import HuggingFaceEmbedder

logger = get_logger(__name__)

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "ragas_tax_cases.json"
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A single TaxBot benchmark question."""

    id: str
    question: str
    ground_truth: str
    expected_doc_numbers: tuple[str, ...]
    expected_tax_year: int | None


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The output of running one case through the live pipeline."""

    case: EvaluationCase
    answer: str
    contexts: tuple[str, ...]
    failure_reason: str | None


def load_cases(path: Path = DEFAULT_FIXTURE_PATH) -> list[EvaluationCase]:
    """Load and validate the benchmark fixture."""

    if not path.is_file():
        raise FileNotFoundError(f"Benchmark fixture missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if len(raw_cases) < 20:
        raise ValueError(
            f"Benchmark fixture must contain >=20 cases, found {len(raw_cases)}"
        )
    cases: list[EvaluationCase] = []
    for item in raw_cases:
        cases.append(
            EvaluationCase(
                id=str(item["id"]),
                question=str(item["question"]),
                ground_truth=str(item["ground_truth"]),
                expected_doc_numbers=tuple(item.get("expected_doc_numbers", [])),
                expected_tax_year=(
                    int(item["expected_tax_year"])
                    if item.get("expected_tax_year") is not None
                    else None
                ),
            )
        )
    return cases


async def run_pipeline(
    cases: list[EvaluationCase],
    *,
    settings: Settings | None = None,
    max_concurrency: int = 4,
) -> list[CaseResult]:
    """Execute each case through the live TaxBot generator."""

    settings = settings or get_settings()
    results: list[CaseResult] = []
    semaphore = asyncio.Semaphore(max_concurrency)

    async with Database(settings) as database, HuggingFaceEmbedder(settings) as embedder:
        repository = DocumentRepository(database)
        retriever = HybridRetriever(repository, embedder, settings=settings)
        generator = AnswerGenerator(retriever, settings=settings)

        async def _one(case: EvaluationCase) -> CaseResult:
            async with semaphore:
                try:
                    result, context = await generator.answer_with_context(case.question)
                    contexts = tuple(parent.text_content for parent in context.parent_nodes)
                    return CaseResult(
                        case=case,
                        answer=result.answer,
                        contexts=contexts,
                        failure_reason=None,
                    )
                except Exception as exc:
                    logger.warning(
                        "evaluation_case_failed",
                        case_id=case.id,
                        error=str(exc),
                    )
                    return CaseResult(
                        case=case,
                        answer="",
                        contexts=(),
                        failure_reason=str(exc),
                    )

        results.extend(await asyncio.gather(*(_one(case) for case in cases)))

    return results


def _build_ragas_dataset(results: list[CaseResult]):  # type: ignore[no-untyped-def]
    """Convert pipeline outputs to a Ragas-compatible dataset."""

    from datasets import Dataset

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for record in results:
        if record.failure_reason is not None or not record.answer:
            continue
        questions.append(record.case.question)
        answers.append(record.answer)
        contexts.append(list(record.contexts) or [""])
        ground_truths.append(record.case.ground_truth)

    if not questions:
        raise RuntimeError("All evaluation cases failed before reaching Ragas")

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def _make_judge_llm(settings: Settings):  # type: ignore[no-untyped-def]
    """Build the LLM that Ragas uses internally for grading."""

    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key.get_secret_value(),
            temperature=0.0,
        )
    )


def evaluate_results(results: list[CaseResult], settings: Settings) -> dict[str, float]:
    """Run Ragas metrics and return the mean score per metric."""

    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = _build_ragas_dataset(results)
    judge_llm = _make_judge_llm(settings)
    evaluation = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        raise_exceptions=False,
    )

    scores = evaluation.to_pandas()
    metric_means: dict[str, float] = {}
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        if metric in scores.columns:
            metric_means[metric] = float(scores[metric].dropna().mean())
        else:
            metric_means[metric] = math.nan

    return metric_means


def enforce_thresholds(
    metrics: Mapping[str, float],
    settings: Settings,
) -> list[str]:
    """Return a list of human-readable threshold violation messages."""

    expectations = {
        "faithfulness": settings.faithfulness_threshold,
        "answer_relevancy": settings.answer_relevancy_threshold,
        "context_precision": settings.context_precision_threshold,
        "context_recall": settings.context_recall_threshold,
    }
    failures: list[str] = []
    for metric, threshold in expectations.items():
        observed = metrics.get(metric, math.nan)
        if math.isnan(observed):
            failures.append(f"{metric}: missing score (judge LLM produced no value)")
            continue
        if observed <= threshold:
            failures.append(
                f"{metric}: {observed:.4f} <= threshold {threshold:.4f}"
            )
    return failures


async def run_evaluation(
    *,
    settings: Settings | None = None,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    max_concurrency: int = 4,
) -> dict[str, Any]:
    settings = settings or get_settings()
    cases = load_cases(fixture_path)
    results = await run_pipeline(
        cases,
        settings=settings,
        max_concurrency=max_concurrency,
    )

    pipeline_failures = [r for r in results if r.failure_reason is not None]
    metrics = evaluate_results(results, settings)
    threshold_failures = enforce_thresholds(metrics, settings)

    summary = {
        "total_cases": len(cases),
        "pipeline_failures": len(pipeline_failures),
        "scored_cases": len(cases) - len(pipeline_failures),
        "metrics": metrics,
        "threshold_failures": threshold_failures,
    }
    logger.info("evaluation_summary", **summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TaxBot Ragas evaluation gate")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to the JSON benchmark fixture.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max simultaneous case executions through the live pipeline.",
    )
    return parser.parse_args()


def main() -> NoReturn:
    args = _parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)
    summary = asyncio.run(
        run_evaluation(
            settings=settings,
            fixture_path=args.fixture,
            max_concurrency=args.concurrency,
        )
    )
    threshold_failures: list[str] = list(summary["threshold_failures"])
    if threshold_failures:
        logger.error("evaluation_failed", failures=threshold_failures)
        sys.stderr.write("\n".join(threshold_failures) + "\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
