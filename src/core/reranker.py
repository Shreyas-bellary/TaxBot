"""Cross-encoder reranker backed by a fine-tuned LoRA adapter.

Loads once at startup. All torch inference runs in the default thread-pool
executor so the async event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import functools
import json
from pathlib import Path

from core.logging_config import get_logger

logger = get_logger(__name__)

_MAX_LENGTH = 512


class ChildReranker:
    """Score query-passage pairs with a LoRA-adapted cross-encoder.

    Args:
        model_path: Directory containing ``adapter_config.json``,
                    ``adapter_model.safetensors``, and the tokenizer files.
                    The base model name is read from ``adapter_config.json``.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        hf_token: str | None = None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = Path(model_path)
        adapter_cfg = model_path / "adapter_config.json"
        if not adapter_cfg.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found at {model_path}. "
                "Check TAXBOT_RERANKER_MODEL_PATH."
            )

        base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        logger.info("reranker_loading", base=base_name, adapter=str(model_path))

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            token=hf_token,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            base_name,
            num_labels=1,
            token=hf_token,
        )
        self._model = PeftModel.from_pretrained(
            base,
            str(model_path),
            token=hf_token,
        )
        self._model.eval()

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        logger.info("reranker_ready", device=self._device)

    def _score_batch(self, query: str, passages: list[str]) -> list[float]:
        """Score query-passage pairs. Runs synchronously; call via executor."""
        import torch

        if not passages:
            return []

        inputs = self._tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, return_dict=True)
            logits = outputs.logits
            if logits.dim() == 2:
                logits = logits.squeeze(-1)

        scores = logits.cpu().tolist()
        return scores if isinstance(scores, list) else [scores]

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score per passage (higher = more relevant)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self._score_batch, query, passages),
        )
