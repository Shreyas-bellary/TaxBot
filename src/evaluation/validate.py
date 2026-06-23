"""Ragas evaluation gate.

Runs the 20-case TaxBot benchmark, computes Faithfulness, Answer Relevance,
Context Precision, and Context Recall, and asserts each metric exceeds the
threshold configured in :class:`Settings`. If any metric falls below the
gate, the process exits with a non-zero status so CI fails immediately.

The evaluator runs the live TaxBot pipeline end-to-end and uses the
Ragas judge LLM configured via ``eval_judge_provider`` / ``eval_judge_model``
in :class:`Settings`. Network and database calls are wrapped in
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
from typing import TYPE_CHECKING, Any, NoReturn
from uuid import UUID

from pydantic import HttpUrl

from core.config import Settings, get_settings
from core.db import Database
from core.errors import OutputCitationError
from core.generation import AnswerGenerator
from core.logging_config import configure_logging, get_logger
from core.models import GenerationResult, RetrievedContext
from core.repository import DocumentRepository
from core.reranker import ChildReranker
from core.retrieval import HybridRetriever
from core.security import OutputGuard
from core.vector_store import QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder
from ingestion.sparse_encoder import SparseEncoder

if TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.outputs import LLMResult
    from langchain_core.prompt_values import PromptValue
    from ragas.llms.base import BaseRagasLLM

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


class _EvalOutputGuard(OutputGuard):
    """Allow uncited answers through so Ragas can score retrieval quality."""

    def validate(
        self,
        *,
        answer: str,
        context: RetrievedContext,
    ) -> GenerationResult:
        if not isinstance(answer, str) or not answer.strip():
            raise OutputCitationError("Empty LLM completion")

        if self._settings.user_query_start_tag in answer or self._settings.user_query_end_tag in answer:
            raise OutputCitationError("Completion leaked the user-query fence tags")

        cited_urls: list[HttpUrl] = []
        used_parents: list[UUID] = []
        for parent in context.parent_nodes:
            source_url = parent.metadata.get("source_url")
            if not isinstance(source_url, str):
                continue
            if source_url in answer:
                cited_urls.append(HttpUrl(source_url))
                used_parents.append(parent.id)

        return GenerationResult(
            answer=answer.strip(),
            citations=tuple(cited_urls),
            used_parent_ids=tuple(used_parents),
        )


def _create_gemini_ragas_llm(*, model_id: str, api_key: str) -> BaseRagasLLM:
    """Build a Ragas judge that calls google-genai directly."""

    from google import genai
    from google.genai import types as genai_types
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=4096,
        response_mime_type="text/plain",
    )

    class _GeminiRagasLLM(BaseRagasLLM):  # type: ignore[misc]
        def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            del n, temperature, stop, callbacks
            response = client.models.generate_content(
                model=model_id,
                contents=_prompt_to_text(prompt),
                config=config,
            )
            text = (getattr(response, "text", "") or "").strip()
            return LLMResult(generations=[[Generation(text=text)]])

        async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float | None = None,
            stop: list[str] | None = None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            del n, temperature, stop, callbacks
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=_prompt_to_text(prompt),
                config=config,
            )
            text = (getattr(response, "text", "") or "").strip()
            return LLMResult(generations=[[Generation(text=text)]])

    return _GeminiRagasLLM()


def _prompt_to_text(prompt: PromptValue) -> str:
    to_string = getattr(prompt, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(prompt)


def _make_reranker(settings: Settings) -> ChildReranker | None:
    if not settings.reranker_enabled:
        return None
    return ChildReranker(
        settings.reranker_model_path,
        hf_token=settings.huggingface_api_token.get_secret_value(),
    )


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
        vector_store = QdrantVectorStore(settings)
        await vector_store.ensure_collection()
        sparse_encoder = SparseEncoder(settings)
        repository = DocumentRepository(database)
        retriever = HybridRetriever(
            repository,
            embedder,
            vector_store,
            sparse_encoder,
            reranker=_make_reranker(settings),
            settings=settings,
        )
        generator = AnswerGenerator(
            retriever,
            settings=settings,
            output_guard=_EvalOutputGuard(settings),
        )

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

    if settings.eval_judge_provider == "gemini":
        return _create_gemini_ragas_llm(
            model_id=settings.eval_judge_model,
            api_key=settings.gemini_api_key.get_secret_value(),
        )

    if settings.eval_judge_provider == "openrouter":
        if settings.openrouter_api_key is None:
            raise ValueError(
                "TAXBOT_OPENROUTER_API_KEY missing but eval_judge_provider=openrouter"
            )
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(
            ChatOpenAI(  # type: ignore[call-arg]
                model=settings.eval_judge_model,
                api_key=settings.openrouter_api_key.get_secret_value(),
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
            )
        )

    raise ValueError(
        f"Unsupported eval_judge_provider for Ragas judge: {settings.eval_judge_provider}"
    )


def _make_judge_embeddings(settings: Settings):  # type: ignore[no-untyped-def]
    """Build embeddings for Ragas metrics that need them (e.g. answer_relevancy)."""

    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        HuggingFaceEndpointEmbeddings(
            model=settings.embedding_model,
            task="feature-extraction",
            huggingfacehub_api_token=settings.huggingface_api_token.get_secret_value(),
        )
    )


@dataclass(frozen=True, slots=True)
class RetrievedContextSummary:
    """Compact retrieval metadata for debug output."""

    doc_number: str
    tax_year: str
    source_url: str
    node_kind: str
    text_preview: str


@dataclass(frozen=True, slots=True)
class CaseInspectResult:
    """Full pipeline output for a single benchmark case."""

    case: EvaluationCase
    answer: str
    citations: tuple[str, ...]
    context_texts: tuple[str, ...]
    retrieved: tuple[RetrievedContextSummary, ...]
    failure_reason: str | None


def find_case(cases: list[EvaluationCase], case_id: str) -> EvaluationCase:
    """Return the case with ``case_id`` or raise with available ids."""

    for case in cases:
        if case.id == case_id:
            return case
    available = ", ".join(case.id for case in cases)
    raise ValueError(f"Unknown case_id {case_id!r}. Available: {available}")


def _context_preview(text: str, *, limit: int = 400) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _summarize_context(context: RetrievedContext) -> tuple[RetrievedContextSummary, ...]:
    summaries: list[RetrievedContextSummary] = []
    for parent in context.parent_nodes:
        metadata = parent.metadata
        summaries.append(
            RetrievedContextSummary(
                doc_number=str(metadata.get("doc_number") or ""),
                tax_year=str(metadata.get("tax_year") or ""),
                source_url=str(metadata.get("source_url") or ""),
                node_kind=str(metadata.get("node_kind") or ""),
                text_preview=_context_preview(parent.text_content),
            )
        )
    return tuple(summaries)


async def inspect_case(
    case: EvaluationCase,
    *,
    settings: Settings | None = None,
) -> CaseInspectResult:
    """Run one case through the live pipeline and capture debug metadata."""

    settings = settings or get_settings()
    async with Database(settings) as database, HuggingFaceEmbedder(settings) as embedder:
        vector_store = QdrantVectorStore(settings)
        await vector_store.ensure_collection()
        sparse_encoder = SparseEncoder(settings)
        repository = DocumentRepository(database)
        retriever = HybridRetriever(
            repository,
            embedder,
            vector_store,
            sparse_encoder,
            reranker=_make_reranker(settings),
            settings=settings,
        )
        generator = AnswerGenerator(
            retriever,
            settings=settings,
            output_guard=_EvalOutputGuard(settings),
        )
        try:
            result, context = await generator.answer_with_context(case.question)
            return CaseInspectResult(
                case=case,
                answer=result.answer,
                citations=tuple(str(url) for url in result.citations),
                context_texts=tuple(parent.text_content for parent in context.parent_nodes),
                retrieved=_summarize_context(context),
                failure_reason=None,
            )
        except Exception as exc:
            logger.warning(
                "evaluation_case_failed",
                case_id=case.id,
                error=str(exc),
            )
            return CaseInspectResult(
                case=case,
                answer="",
                citations=(),
                context_texts=(),
                retrieved=(),
                failure_reason=str(exc),
            )


def _run_ragas_evaluation(results: list[CaseResult], settings: Settings):  # type: ignore[no-untyped-def]
    """Run Ragas and return the raw evaluation object."""

    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = _build_ragas_dataset(results)
    judge_llm = _make_judge_llm(settings)
    return evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=_make_judge_embeddings(settings),
        raise_exceptions=False,
    )


def _metric_means_from_scores(scores) -> dict[str, float]:  # type: ignore[no-untyped-def]
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


def evaluate_results(results: list[CaseResult], settings: Settings) -> dict[str, float]:
    """Run Ragas metrics and return the mean score per metric."""

    evaluation = _run_ragas_evaluation(results, settings)
    return _metric_means_from_scores(evaluation.to_pandas())


def evaluate_case_scores(
    results: list[CaseResult],
    settings: Settings,
) -> dict[str, float]:
    """Return Ragas metric scores for the first (only) scored row."""

    scores = _run_ragas_evaluation(results, settings).to_pandas()
    if scores.empty:
        raise RuntimeError("Ragas returned no scores for the case")
    row = scores.iloc[0]
    case_scores: dict[str, float] = {}
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        if metric in row.index and not math.isnan(float(row[metric])):
            case_scores[metric] = float(row[metric])
    return case_scores


def format_case_debug_report(
    inspection: CaseInspectResult,
    *,
    settings: Settings,
    ragas_scores: Mapping[str, float] | None = None,
) -> str:
    """Render a human-readable debug report for one benchmark case."""

    lines = [
        f"=== Evaluation debug: {inspection.case.id} ===",
        "",
        "QUESTION",
        inspection.case.question,
        "",
        "EXPECTED (ground truth)",
        inspection.case.ground_truth,
        "",
        f"Expected docs: {', '.join(inspection.case.expected_doc_numbers) or '(none)'}",
        f"Expected tax year: {inspection.case.expected_tax_year}",
        "",
    ]

    if inspection.failure_reason is not None:
        lines.extend(
            [
                "PIPELINE STATUS",
                f"FAILED: {inspection.failure_reason}",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "GENERATED ANSWER",
            inspection.answer,
            "",
            f"Citations detected: {', '.join(inspection.citations) or '(none)'}",
            "",
            f"RETRIEVED CONTEXT ({len(inspection.retrieved)} parent nodes)",
        ]
    )
    for index, parent in enumerate(inspection.retrieved, start=1):
        lines.extend(
            [
                f"--- Context {index} ---",
                f"doc_number: {parent.doc_number}",
                f"tax_year: {parent.tax_year}",
                f"node_kind: {parent.node_kind}",
                f"source_url: {parent.source_url}",
                parent.text_preview,
                "",
            ]
        )

    if ragas_scores:
        lines.append("RAGAS SCORES (this case)")
        for metric in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            score = ragas_scores.get(metric)
            threshold = {
                "faithfulness": settings.faithfulness_threshold,
                "answer_relevancy": settings.answer_relevancy_threshold,
                "context_precision": settings.context_precision_threshold,
                "context_recall": settings.context_recall_threshold,
            }[metric]
            if score is None:
                lines.append(f"  {metric}: (missing)")
            else:
                status = "PASS" if score > threshold else "FAIL"
                lines.append(
                    f"  {metric}: {score:.4f} (threshold {threshold:.4f}) [{status}]"
                )
        lines.append("")
        lines.extend(
            [
                "METRIC HINTS",
                "  faithfulness       — is the answer supported by retrieved context?",
                "  answer_relevancy   — does the answer address the question?",
                "  context_precision  — is retrieved context relevant to the question?",
                "  context_recall     — does retrieved context cover the ground truth?",
            ]
        )

    return "\n".join(lines)


async def debug_case(
    case_id: str,
    *,
    settings: Settings | None = None,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
) -> str:
    """Run and format a single-case evaluation debug report."""

    settings = settings or get_settings()
    case = find_case(load_cases(fixture_path), case_id)
    inspection = await inspect_case(case, settings=settings)
    ragas_scores: dict[str, float] | None = None
    if inspection.failure_reason is None:
        ragas_scores = evaluate_case_scores(
            [
                CaseResult(
                    case=inspection.case,
                    answer=inspection.answer,
                    contexts=inspection.context_texts,
                    failure_reason=None,
                )
            ],
            settings,
        )
    return format_case_debug_report(
        inspection,
        settings=settings,
        ragas_scores=ragas_scores,
    )


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
    case_id: str | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    cases = load_cases(fixture_path)
    if case_id is not None:
        cases = [find_case(cases, case_id)]
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
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Run a single benchmark case by id (see tests/fixtures/ragas_tax_cases.json).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a human-readable report for --case-id and exit 0 (skips gate failure).",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print available benchmark case ids and exit.",
    )
    return parser.parse_args()


def _list_case_ids(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> None:
    for case in load_cases(fixture_path):
        sys.stdout.write(f"{case.id}\n")


def main() -> NoReturn:
    args = _parse_args()
    settings = get_settings()

    if args.list_cases:
        _list_case_ids(args.fixture)
        sys.exit(0)

    if args.debug and args.case_id is None:
        sys.stderr.write("--debug requires --case-id\n")
        sys.exit(2)

    configure_logging(level=settings.log_level, as_json=settings.log_json)

    if args.debug:
        report = asyncio.run(
            debug_case(
                args.case_id,
                settings=settings,
                fixture_path=args.fixture,
            )
        )
        sys.stdout.write(report + "\n")
        sys.exit(0)

    summary = asyncio.run(
        run_evaluation(
            settings=settings,
            fixture_path=args.fixture,
            max_concurrency=args.concurrency,
            case_id=args.case_id,
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
