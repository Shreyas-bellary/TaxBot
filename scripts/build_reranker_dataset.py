"""
Build a reranker fine-tuning dataset from TaxBot's hybrid retrieval pipeline.

Algorithm
---------
1. Load 20 existing Q&A pairs from ``tests/fixtures/ragas_tax_cases.json``.
2. Generate 480 additional fact-based tax Q&A pairs via OpenRouter (in batches,
   across 12 rotating topic areas for diversity).
3. For each of the 500 pairs:
   a. Run the live hybrid retriever (BGE dense + BM25/RRF → Qdrant) to fetch the
      top-K **child** nodes (sentence / table-summary chunks), in RRF rank order.
   b. Call an OpenRouter judge that inspects every retrieved child context and
      returns YES/NO for whether it contains or clearly supports the ground-truth
      answer (including table summaries that describe relevant table data).
   c. Emit exactly **one hard positive** (label=1, first YES context by rank)
      and up to **four hard negatives** (label=0, highest-ranked NO contexts).
   d. Skip the pair entirely when no retrieved context contains the answer.
4. Write the labeled dataset to JSON.

Usage (from project root)
--------------------------
    python scripts/build_reranker_dataset.py
    python scripts/build_reranker_dataset.py --output scripts/reranker_training_data.json
    python scripts/build_reranker_dataset.py --reuse-qa   # skip generation; reload saved pairs

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import httpx  # noqa: E402
from tenacity import (  # noqa: E402
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm  # noqa: E402

from core.config import Settings, get_settings  # noqa: E402
from core.db import Database  # noqa: E402
from core.errors import OutOfDomainQueryError, RetrievalError, RouterError  # noqa: E402
from core.logging_config import configure_logging, get_logger  # noqa: E402
from core.query_router import QueryRouteResult, RouteFilters, route_query  # noqa: E402
from core.repository import DocumentRepository  # noqa: E402
from core.retrieval import assess_retrieval_confidence, _encode_query, _normalize_rows  # noqa: E402
from core.security import SanitizedQuery  # noqa: E402
from core.vector_store import QdrantVectorStore  # noqa: E402
from ingestion.embeddings import HuggingFaceEmbedder  # noqa: E402
from ingestion.sparse_encoder import SparseEncoder  # noqa: E402

logger = get_logger(__name__)

# Paths
_FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "ragas_tax_cases.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "scripts" / "reranker_training_data.json"
_DEFAULT_QA_INTERMEDIATE = _PROJECT_ROOT / "scripts" / "qa_pairs_intermediate.json"


# Tunable constants
TARGET_GENERATED_PAIRS: int = 480    # existing 20 + generated 480 = 500 total
GENERATION_BATCH_SIZE: int = 10      # Q&A pairs per single OpenRouter generation call
MAX_HARD_NEGATIVES: int = 4          # max hard negatives emitted per question
RETRIEVAL_CONCURRENCY: int = 2       # simultaneous retriever + judge calls
GENERATION_CONCURRENCY: int = 1      # simultaneous question-generation calls
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503})

# Topic areas rotated across generation batches to maximise diversity.
_TOPIC_AREAS: list[str] = [
    "individual filing requirements, standard deduction amounts, itemized deductions, and filing-status edge cases",
    "tax credits: child tax credit, EITC eligibility, education credits (AOTC/LLC), and phase-out mechanics",
    "retirement accounts: traditional and Roth IRA contribution rules, 401(k) limits, RMDs, and early-distribution penalties",
    "capital gains, qualified dividends, wash-sale rule, Net Investment Income Tax, and collectibles rates",
    "self-employment: Schedule C, SE tax computation, QBI deduction phase-outs, and home-office deduction methods",
    "foreign income: FEIE physical-presence and bona-fide-residence tests, housing exclusion, and foreign tax credit mechanics",
    "business entities: S-corp shareholder basis (Form 7203), partnership at-risk rules, and passive-activity loss limits",
    "depreciation: MACRS conventions, Section 179 limits, bonus depreciation phase-down, and listed-property rules",
    "real estate: rental income reporting, depreciation recapture, 1031 exchange identification rules, and Section 121 partial exclusion",
    "AMT: preference items, AMTI adjustments, exemption amounts, phase-out thresholds, and minimum tax credit carry-forward",
    "estimated taxes, withholding, Form 2210 safe harbors, and underpayment penalty calculation",
    "charitable contributions: AGI limits, noncash substantiation, qualified-appraisal thresholds, QCDs, and donor-advised funds",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class QAPair:
    question: str
    ground_truth: str
    source: str  # "existing" | "generated"


@dataclass
class LabeledSample:
    id: str
    query: str
    context: str
    label: int           # 1 = hard positive, 0 = hard negative
    retrieval_rank: int  # 1-based position in the RRF-ranked child list
    child_id: str
    parent_id: str
    source_url: str
    question_source: str  # "existing" | "generated"


@dataclass
class RetrievedChild:
    """One hybrid-search child hit with Postgres text loaded."""

    child_id: UUID
    parent_id: UUID
    text: str
    source_url: str
    retrieval_rank: int  # 1-based RRF rank


# ---------------------------------------------------------------------------
# OpenRouter LLM client
# ---------------------------------------------------------------------------
@dataclass
class OpenRouterLLM:
    """Thin async wrapper around the OpenRouter chat-completions API."""

    api_key: str
    model: str
    timeout: float = 120.0
    max_retries: int = 6

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "TaxBot",
            },
        ) as client:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_is_openrouter_retryable)
                | retry_if_exception_type(httpx.TransportError),
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                reraise=True,
            ):
                with attempt:
                    response = await client.post(
                        _OPENROUTER_URL,
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    choices = body.get("choices") or []
                    if not choices:
                        raise RuntimeError("OpenRouter returned no choices")
                    content = choices[0].get("message", {}).get("content") or ""
                    if not str(content).strip():
                        raise RuntimeError("OpenRouter returned empty content")
                    return str(content).strip()
        raise RuntimeError("OpenRouter call exhausted retries")  # pragma: no cover


def _is_openrouter_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _OPENROUTER_RETRYABLE_STATUS
    return False


def _make_openrouter_llm(settings: Settings, model: str | None = None) -> OpenRouterLLM:
    if settings.openrouter_api_key is None:
        raise ValueError(
            "TAXBOT_OPENROUTER_API_KEY is required. Set it in .env or export it."
        )
    return OpenRouterLLM(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=model or settings.openrouter_model,
        timeout=settings.irs_request_timeout_seconds,
        max_retries=settings.gemini_max_retries,
    )


def _strip_json_fences(text: str) -> str:
    """Remove accidental markdown fences some models emit around JSON."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [line for line in stripped.splitlines() if not line.startswith("```")]
    return "\n".join(lines).strip()


class ChildHybridRetriever:
    """Hybrid retriever that returns top-K child nodes in RRF order."""

    def __init__(
        self,
        repository: DocumentRepository,
        embedder: HuggingFaceEmbedder,
        vector_store: QdrantVectorStore,
        sparse_encoder: SparseEncoder,
        *,
        settings: Settings,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_encoder = sparse_encoder
        self._settings = settings

    async def retrieve_children(
        self,
        query: str,
        *,
        sanitized: SanitizedQuery | None = None,
    ) -> list[RetrievedChild]:
        """Run hybrid search and return top-K child chunks with text loaded."""
        if not query or not query.strip():
            raise RetrievalError("Empty query")

        top_k = self._settings.retrieval_top_k_children

        if sanitized is not None:
            san = sanitized
        else:
            start = self._settings.user_query_start_tag
            end = self._settings.user_query_end_tag
            san = SanitizedQuery(
                cleaned_text=query,
                fenced_prompt_section=f"[{start}]\n{query}\n[{end}]",
                start_tag=start,
                end_tag=end,
            )

        route: QueryRouteResult | None = None
        try:
            route = await route_query(san, settings=self._settings)
        except OutOfDomainQueryError:
            raise
        except RouterError as exc:
            logger.warning(
                "query_router_fallback",
                error=str(exc),
                query_preview=query[:120],
            )
            route = None

        filters: RouteFilters = route.filters if route else RouteFilters()

        dense_embedding, sparse_vector = await _encode_query(
            query, self._embedder, self._sparse_encoder
        )

        has_filters = bool(
            filters.tax_year is not None or filters.doc_type is not None
        )

        results = await self._vector_store.hybrid_search(
            dense_vector=dense_embedding,
            sparse_vector=sparse_vector,
            top_k=top_k,
            tax_year=filters.tax_year,
            doc_type=filters.doc_type,
        )

        if not results and has_filters:
            results = await self._vector_store.hybrid_search(
                dense_vector=dense_embedding,
                sparse_vector=sparse_vector,
                top_k=top_k,
            )

        if not results:
            raise RetrievalError("No matching child nodes for query")

        rows = _normalize_rows(results)
        assess_retrieval_confidence(
            rows,
            settings=self._settings,
            query_preview=query[:120],
        )

        child_ids = [UUID(str(row["child_id"])) for row in rows]
        parent_ids = list(dict.fromkeys(UUID(str(row["parent_id"])) for row in rows))

        child_texts, parent_records = await asyncio.gather(
            self._repo.fetch_children_text(child_ids),
            self._repo.fetch_parents(parent_ids),
        )

        parent_urls: dict[UUID, str] = {}
        for pid, record in parent_records.items():
            url = record["metadata"].get("source_url")
            parent_urls[pid] = str(url) if isinstance(url, str) else ""

        children: list[RetrievedChild] = []
        for rank, row in enumerate(rows, start=1):
            cid = UUID(str(row["child_id"]))
            pid = UUID(str(row["parent_id"]))
            text = child_texts.get(cid)
            if not text:
                continue
            children.append(
                RetrievedChild(
                    child_id=cid,
                    parent_id=pid,
                    text=text,
                    source_url=parent_urls.get(pid, ""),
                    retrieval_rank=rank,
                )
            )

        if not children:
            raise RetrievalError("Child node text missing for hybrid search hits")

        return children


# ---------------------------------------------------------------------------
# 1. Load existing cases
# ---------------------------------------------------------------------------
def load_existing_cases() -> list[QAPair]:
    """Load the 20 benchmark Q&A pairs from the Ragas fixture file."""
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return [
        QAPair(
            question=item["question"],
            ground_truth=item["ground_truth"],
            source="existing",
        )
        for item in payload["cases"]
    ]


# ---------------------------------------------------------------------------
# 2. Question generation
# ---------------------------------------------------------------------------
def _generation_prompt(batch_size: int, topic: str, existing_sample: list[str]) -> str:
    exclusions = "\n".join(f"  - {q}" for q in existing_sample[:25])
    return textwrap.dedent(f"""
        You are a US federal tax expert with authoritative knowledge of current IRS
        publications, form instructions, revenue procedures, and regulations.

        Generate exactly {batch_size} unique, fact-based US federal income tax questions
        focused on this topic area: **{topic}**

        REQUIREMENTS:
        1. Each question must target a specific rule, dollar threshold, percentage, form
           field, safe-harbor condition, or multi-step procedural requirement.
        2. Every ground_truth must be 100% accurate with exact figures — no approximations.
        3. Do NOT overlap with or paraphrase any existing question listed below:
        {exclusions}
        4. State the applicable tax year in the question when citing year-specific amounts and target 2025 and 2026 only.
        5. Answers must be complete, self-contained sentences (50-100 words each).
        6. Mix complexity: include both straightforward threshold lookups and nuanced
           coordination / phase-out / ordering rules.

        Respond with ONLY a valid JSON array of exactly {batch_size} objects:
        [{{"question": "...", "ground_truth": "..."}}, ...]
        No other text before or after the array.
    """).strip()


async def _generate_one_batch(
    llm: OpenRouterLLM,
    batch_size: int,
    topic: str,
    existing_questions: list[str],
) -> list[QAPair]:
    """Call OpenRouter once to produce ``batch_size`` Q&A pairs for ``topic``."""
    prompt = _generation_prompt(batch_size, topic, existing_questions)
    raw = await llm.complete(prompt, temperature=0.8, max_tokens=8192)
    items: list[dict[str, str]] = json.loads(_strip_json_fences(raw))

    if not isinstance(items, list):
        raise ValueError(f"Generation returned non-list: {type(items).__name__}")

    pairs: list[QAPair] = []
    for item in items:
        q = str(item.get("question", "")).strip()
        gt = str(item.get("ground_truth", "")).strip()
        if len(q) > 20 and len(gt) > 20:
            pairs.append(QAPair(question=q, ground_truth=gt, source="generated"))
    return pairs


async def generate_qa_pairs(
    llm: OpenRouterLLM,
    target: int,
    existing_questions: list[str],
) -> list[QAPair]:
    """
    Generate ``target`` Q&A pairs across rotating topic areas.

    Batches run with ``GENERATION_CONCURRENCY`` parallelism. Each failed batch
    is retried up to 3 times before being silently dropped.
    """
    num_batches = (target + GENERATION_BATCH_SIZE - 1) // GENERATION_BATCH_SIZE
    batch_sizes = [GENERATION_BATCH_SIZE] * num_batches
    if remainder := target % GENERATION_BATCH_SIZE:
        batch_sizes[-1] = remainder
    topics = [_TOPIC_AREAS[i % len(_TOPIC_AREAS)] for i in range(num_batches)]

    sem = asyncio.Semaphore(GENERATION_CONCURRENCY)
    all_pairs: list[QAPair] = []

    async def _with_retry(idx: int) -> list[QAPair]:
        async with sem:
            for backoff in (2, 4, 8):
                try:
                    return await _generate_one_batch(
                        llm, batch_sizes[idx], topics[idx], existing_questions
                    )
                except Exception as exc:
                    logger.warning(
                        "generation_batch_failed",
                        batch=idx,
                        topic=topics[idx],
                        error=str(exc),
                    )
                    await asyncio.sleep(backoff)
            return []

    tasks = [asyncio.create_task(_with_retry(i)) for i in range(num_batches)]

    with tqdm(total=target, desc="Generating Q&A pairs", unit="q", ncols=90) as pbar:
        for fut in asyncio.as_completed(tasks):
            batch = await fut
            all_pairs.extend(batch)
            pbar.update(len(batch))

    # Deduplicate against existing questions and within the generated set.
    seen: set[str] = {q.lower().strip() for q in existing_questions}
    unique: list[QAPair] = []
    for pair in all_pairs:
        key = pair.question.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    return unique[:target]


# ---------------------------------------------------------------------------
# 3. LLM grounding judge
# ---------------------------------------------------------------------------
def _grounding_prompt(question: str, ground_truth: str, contexts: list[str]) -> str:
    blocks = "\n\n".join(
        f"[Context {i + 1}]\n{text[:1800].strip()}"
        for i, text in enumerate(contexts)
    )
    n = len(contexts)
    return textwrap.dedent(f"""
        You are a tax-law accuracy judge. For each numbered context passage below,
        determine whether it DIRECTLY CONTAINS or CLEARLY SUPPORTS the correct answer
        to the given tax question.

        Question: {question}

        Correct Answer: {ground_truth}

        {blocks}

        Judging rules:
        1. Reply "YES" if the passage contains enough information to derive or
           confirm the correct answer on its own.
        2. TABLE SUMMARIES: Some contexts are short 3-sentence summaries of an IRS
           table, not the full table. If a summary states or clearly describes that
           a table contains the data needed to answer the question — e.g. it names
           the topic, axes, thresholds, or takeaways covered by the table — reply
           "YES" even when the exact figures from the correct answer are not
           reproduced verbatim in the summary. Reply "NO" only when the summary does
           not refer to or describe the relevant table topic at all.
        3. Reply "NO" if the passage is off-topic or cannot reasonably support the
           correct answer.

        For each context 1 through {n}, reply "YES" or "NO".

        Respond with ONLY a JSON array of exactly {n} strings, one per context in order.
        Example (for {n} contexts): {json.dumps(["YES"] + ["NO"] * (n - 1))}
        No other text.
    """).strip()


async def check_grounding(
    llm: OpenRouterLLM,
    question: str,
    ground_truth: str,
    contexts: list[str],
) -> list[bool]:
    """
    Return a bool per context: True if it directly contains/supports the answer.

    Table-summary child nodes count as grounded when they describe the relevant
    table data even if exact figures are not inlined in the summary text.

    On parse failure the function raises so the caller can skip the pair.
    """
    if not contexts:
        return []

    prompt = _grounding_prompt(question, ground_truth, contexts)
    raw = await llm.complete(prompt, temperature=0.0, max_tokens=2048)
    parsed: list[str] = json.loads(_strip_json_fences(raw))

    if not isinstance(parsed, list):
        raise ValueError(f"Grounding judge returned {type(parsed).__name__}, expected list")

    results = [str(v).strip().upper() == "YES" for v in parsed[: len(contexts)]]
    # Pad with False if the model returned fewer entries than contexts.
    results += [False] * (len(contexts) - len(results))
    return results


# ---------------------------------------------------------------------------
# 4. Core pair processor
# ---------------------------------------------------------------------------
async def process_pair(
    pair: QAPair,
    retriever: ChildHybridRetriever,
    llm: OpenRouterLLM,
    semaphore: asyncio.Semaphore,
) -> list[LabeledSample]:
    """
    Retrieve and label one Q&A pair.

    Steps:
      1. Call the hybrid retriever → top-K RRF-ranked child nodes.
      2. Call the OpenRouter judge on all retrieved child contexts.
      3. Select the first YES context as the hard positive.
      4. Select the top ``MAX_HARD_NEGATIVES`` NO contexts (by retrieval rank)
         as hard negatives.

    Returns an empty list when no retrieved context contains the answer
    (pair is silently skipped by the caller).
    """
    async with semaphore:
        # -- Retrieval -------------------------------------------------------
        try:
            children = await retriever.retrieve_children(pair.question)
        except (OutOfDomainQueryError, RetrievalError) as exc:
            logger.debug("pair_skipped", reason=str(exc), q=pair.question[:80])
            return []
        except Exception as exc:
            logger.warning("pair_retrieval_error", error=str(exc), q=pair.question[:80])
            return []

        if not children:
            return []

        texts = [child.text for child in children]

        # -- Grounding check -------------------------------------------------
        try:
            grounded = await check_grounding(
                llm, pair.question, pair.ground_truth, texts
            )
        except Exception as exc:
            logger.warning("pair_grounding_error", error=str(exc), q=pair.question[:80])
            return []

        # -- Hard positive: first context (by retrieval rank) that is grounded
        positive_idx: int | None = next(
            (i for i, g in enumerate(grounded) if g), None
        )
        if positive_idx is None:
            logger.debug("pair_no_positive", q=pair.question[:80])
            return []

        # -- Hard negatives: top-ranked non-grounded contexts ----------------
        neg_indices = [
            i for i, g in enumerate(grounded) if not g
        ][:MAX_HARD_NEGATIVES]

        # -- Assemble labeled samples ----------------------------------------
        samples: list[LabeledSample] = [
            LabeledSample(
                id=str(uuid.uuid4()),
                query=pair.question,
                context=children[positive_idx].text,
                label=1,
                retrieval_rank=children[positive_idx].retrieval_rank,
                child_id=str(children[positive_idx].child_id),
                parent_id=str(children[positive_idx].parent_id),
                source_url=children[positive_idx].source_url,
                question_source=pair.source,
            )
        ]
        for neg_idx in neg_indices:
            child = children[neg_idx]
            samples.append(
                LabeledSample(
                    id=str(uuid.uuid4()),
                    query=pair.question,
                    context=child.text,
                    label=0,
                    retrieval_rank=child.retrieval_rank,
                    child_id=str(child.child_id),
                    parent_id=str(child.parent_id),
                    source_url=child.source_url,
                    question_source=pair.source,
                )
            )
        return samples


# ---------------------------------------------------------------------------
# 5. Dataset builder
# ---------------------------------------------------------------------------
async def build_dataset(
    qa_pairs: list[QAPair],
    settings: Settings,
    output_path: Path,
    llm: OpenRouterLLM,
) -> dict[str, Any]:
    """
    Wire up the live infrastructure, run all pairs concurrently, and write output.

    Returns the stats dict embedded in the output JSON.
    """
    semaphore = asyncio.Semaphore(RETRIEVAL_CONCURRENCY)

    all_samples: list[LabeledSample] = []
    skipped = 0

    async with Database(settings) as db, HuggingFaceEmbedder(settings) as embedder:
        vector_store = QdrantVectorStore(settings)
        await vector_store.ensure_collection()
        sparse_encoder = SparseEncoder(settings)
        repository = DocumentRepository(db)
        retriever = ChildHybridRetriever(
            repository, embedder, vector_store, sparse_encoder, settings=settings
        )

        tasks = [
            asyncio.create_task(
                process_pair(pair, retriever, llm, semaphore)
            )
            for pair in qa_pairs
        ]

        with tqdm(
            total=len(qa_pairs),
            desc="Retrieving & labeling",
            unit="pair",
            ncols=90,
        ) as pbar:
            for fut in asyncio.as_completed(tasks):
                samples = await fut
                if samples:
                    all_samples.extend(samples)
                else:
                    skipped += 1
                pbar.update(1)
                pbar.set_postfix(
                    {"samples": len(all_samples), "skipped": skipped}, refresh=False
                )

    n_pos = sum(1 for s in all_samples if s.label == 1)
    n_neg = sum(1 for s in all_samples if s.label == 0)
    successful = len(qa_pairs) - skipped

    output: dict[str, Any] = {
        "version": "1.0",
        "description": (
            "Reranker fine-tuning dataset derived from TaxBot hybrid retrieval "
            "(Qdrant BGE-large-en-v1.5 dense + BM25/RRF) over IRS child nodes. "
            "Each context is a child-node text_summary (sentence or table summary). "
            "Hard positives are LLM-verified grounded contexts (including table "
            "summaries that describe relevant table data); hard negatives are "
            "the highest-ranked retrieved child contexts that do not contain the ground-truth."
        ),
        "stats": {
            "total_qa_pairs": len(qa_pairs),
            "successful_pairs": successful,
            "skipped_pairs_no_positive": skipped,
            "total_labeled_samples": len(all_samples),
            "hard_positives": n_pos,
            "hard_negatives": n_neg,
        },
        "samples": [
            {
                "id": s.id,
                "query": s.query,
                "context": s.context,
                "label": s.label,
                "retrieval_rank": s.retrieval_rank,
                "child_id": s.child_id,
                "parent_id": s.parent_id,
                "source_url": s.source_url,
                "question_source": s.question_source,
            }
            for s in all_samples
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output["stats"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def _async_main(args: argparse.Namespace) -> None:
    settings = get_settings()
    # Suppress verbose structured logs from retriever / embedder so tqdm output
    # stays readable. Warnings and above are still shown.
    configure_logging(level="WARNING", as_json=False)

    # -- Step 1: load existing 20 pairs --------------------------------------
    existing = load_existing_cases()
    print(f"Loaded {len(existing)} existing Q&A pairs from fixture.")

    # -- Step 2: load or generate 480 new pairs ------------------------------
    qa_intermediate: Path = args.qa_intermediate

    if args.reuse_qa and qa_intermediate.is_file():
        print(f"Reusing intermediate Q&A pairs from {qa_intermediate}")
        raw_items: list[dict[str, str]] = json.loads(
            qa_intermediate.read_text(encoding="utf-8")
        )
        generated = [
            QAPair(
                question=item["question"],
                ground_truth=item["ground_truth"],
                source=item.get("source", "generated"),
            )
            for item in raw_items
            if item.get("source") == "generated"
        ]
        print(f"  Reloaded {len(generated)} generated pairs.")
    else:
        llm = _make_openrouter_llm(settings, args.model)
        existing_questions = [p.question for p in existing]

        print(
            f"\nGenerating {TARGET_GENERATED_PAIRS} new Q&A pairs "
            f"via OpenRouter ({llm.model})..."
        )
        generated = await generate_qa_pairs(
            llm,
            TARGET_GENERATED_PAIRS,
            existing_questions,
        )
        print(f"  Generated {len(generated)} unique pairs after deduplication.")

        # Persist the combined pool so a crash in phase 3 doesn't require
        # re-spending generation tokens.
        all_for_intermediate = existing + generated
        qa_intermediate.parent.mkdir(parents=True, exist_ok=True)
        qa_intermediate.write_text(
            json.dumps(
                [
                    {
                        "question": p.question,
                        "ground_truth": p.ground_truth,
                        "source": p.source,
                    }
                    for p in all_for_intermediate
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  Intermediate Q&A pool saved → {qa_intermediate}")

    all_pairs = existing + generated
    print(f"\nTotal Q&A pairs to process: {len(all_pairs)}")

    llm = _make_openrouter_llm(settings, args.model)
    print(f"OpenRouter model: {llm.model}")

    # -- Step 3: retrieve, judge, label --------------------------------------
    print(
        f"\nRunning hybrid retrieval + grounding judge "
        f"(concurrency={RETRIEVAL_CONCURRENCY}, "
        f"max_negatives={MAX_HARD_NEGATIVES})..."
    )
    stats = await build_dataset(all_pairs, settings, args.output, llm)

    # -- Step 4: summary -----------------------------------------------------
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Dataset written → {args.output}")
    print(f"  {sep[2:]}")
    print(f"  Total labeled samples : {stats['total_labeled_samples']:>6}")
    print(f"    Hard positives       : {stats['hard_positives']:>6}")
    print(f"    Hard negatives       : {stats['hard_negatives']:>6}")
    print(f"  Q&A pairs processed   : {stats['total_qa_pairs']:>6}")
    print(f"    Successful           : {stats['successful_pairs']:>6}")
    print(f"    Skipped (no positive): {stats['skipped_pairs_no_positive']:>6}")
    print(sep)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reranker fine-tuning dataset from TaxBot hybrid retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="PATH",
        help="Path to write the labeled JSON dataset.",
    )
    parser.add_argument(
        "--qa-intermediate",
        type=Path,
        default=_DEFAULT_QA_INTERMEDIATE,
        metavar="PATH",
        help=(
            "Path to save (or reload with --reuse-qa) the intermediate Q&A pool. "
            "Enables crash recovery without re-spending generation tokens."
        ),
    )
    parser.add_argument(
        "--reuse-qa",
        action="store_true",
        default=False,
        help=(
            "Skip question generation and reload the Q&A pool from --qa-intermediate. "
            "Useful for re-running the retrieval phase with different settings."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "OpenRouter model id (defaults to TAXBOT_OPENROUTER_MODEL). "
            "Example: google/gemini-2.0-flash"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=RETRIEVAL_CONCURRENCY,
        metavar="N",
        help="Maximum simultaneous retriever + judge coroutines.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Allow overriding the module-level constant via CLI flag.
    global RETRIEVAL_CONCURRENCY
    RETRIEVAL_CONCURRENCY = args.concurrency
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
