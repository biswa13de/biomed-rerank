"""Cross-encoder reranking — Task 3.

Unlike the bi-encoder in ``src/retrieval/dense.py``, which embeds the query and each
document *independently* and compares the resulting vectors, a cross-encoder feeds the
query and a candidate document into the transformer **together**, as a single sequence
``[CLS] query [SEP] document [SEP]``. Every layer's self-attention can then attend
across the query/document boundary, so the model reasons about *this specific pair* —
it can weigh a document token against a query token directly, something two
independently-produced embeddings can never do after the fact. That joint attention is
also why a cross-encoder cannot be a stage-1 retriever: scoring one pair costs a full
forward pass, so scoring all N corpus documents costs O(N) forward passes per query.
Reranking only ever runs it over the top-K candidates a cheap stage-1 method already
narrowed down (see docs/ARCHITECTURE.md §5, Figure 2).

General Instruction 9 forbids opaque use of built-in APIs, so this deliberately avoids
``sentence_transformers.CrossEncoder.predict()`` and instead calls
``AutoTokenizer``/``AutoModelForSequenceClassification`` directly, extracting the logit
by hand — see :func:`CrossEncoder.explain_one` for the worked single-pair trace the
brief asks for.

Uses the ``text_raw`` view (see src/data_prep.py) for the same reason the dense
retriever does: the transformer was pretrained on natural, cased text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from src.config import load_config, resolve_device


@dataclass
class RerankedDoc:
    doc_id: str
    score: float
    n_tokens_before_truncation: int
    was_truncated: bool


@dataclass
class PairTrace:
    """Everything needed to show one query-document pair's manual scoring path."""

    query: str
    doc_id: str
    input_ids: list[int]
    tokens: list[str]
    n_tokens_before_truncation: int
    was_truncated: bool
    logit: float


class CrossEncoder:
    """Manually-tokenized cross-encoder reranker over a fixed (query, candidates) pair.

    Batch size, max length, and truncation strategy are read from
    ``config/experiment.yaml`` so they stay identical across every model and pool-size
    configuration in Task 6 — the brief's Experimental Constraint clause.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = load_config()
        self.model_name = model_name or cfg.get("rerank.models.primary")
        self.device = device or resolve_device(cfg.get("rerank.device", "auto"))
        self.batch_size = batch_size or cfg.get("rerank.batch_size", 16)
        self.max_length = max_length or cfg.get("rerank.max_length", 512)
        self.truncation = cfg.get("rerank.truncation", "longest_first")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _n_tokens_before_truncation(self, query: str, doc_text: str) -> int:
        """Untruncated pair length, used only to detect and log truncation.

        Tokenizing twice (once here, once truncated inside :meth:`rerank`) is
        deliberately wasteful of a few milliseconds in exchange for an honest
        ``was_truncated`` flag — Task 7's error analysis depends on this being
        measured, not guessed.
        """
        encoded = self.tokenizer(query, doc_text, truncation=False)
        return len(encoded["input_ids"])

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[RerankedDoc]:
        """Score every ``(doc_id, doc_text)`` candidate against ``query``, sorted desc.

        Runs in fixed-size batches under ``torch.no_grad()`` (no gradient tracking
        needed at inference) with the model in ``eval()`` mode (disables dropout) — both
        set once in ``__init__``/the decorator rather than per-call, but stated here
        since they are exactly the two things that make inference numbers reproducible.
        """
        results: list[RerankedDoc] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            doc_texts = [text for _, text in batch]
            queries = [query] * len(batch)

            encoded = self.tokenizer(
                queries,
                doc_texts,
                padding=True,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits.squeeze(-1)  # (batch,) — 1 label
            scores = logits.detach().cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]

            for (doc_id, doc_text), score in zip(batch, scores):
                n_before = self._n_tokens_before_truncation(query, doc_text)
                results.append(
                    RerankedDoc(
                        doc_id=doc_id,
                        score=float(score),
                        n_tokens_before_truncation=n_before,
                        was_truncated=n_before > self.max_length,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k] if top_k is not None else results

    def explain_one(self, query: str, doc_id: str, doc_text: str) -> PairTrace:
        """Manually tokenize and score one pair, exposing every intermediate value.

        This is the worked single-query trace the brief asks for (ARCHITECTURE.md
        requirement 5.3): the constructed ``[CLS] query [SEP] doc [SEP]`` sequence,
        its ``input_ids``, and the logit extracted by hand rather than through
        ``CrossEncoder.predict()``.
        """
        n_before = self._n_tokens_before_truncation(query, doc_text)
        encoded = self.tokenizer(
            query,
            doc_text,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logit = self.model(**encoded).logits.squeeze().item()

        input_ids = encoded["input_ids"][0].tolist()
        return PairTrace(
            query=query,
            doc_id=doc_id,
            input_ids=input_ids,
            tokens=self.tokenizer.convert_ids_to_tokens(input_ids),
            n_tokens_before_truncation=n_before,
            was_truncated=n_before > self.max_length,
            logit=float(logit),
        )
