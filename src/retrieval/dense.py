"""Dense (bi-encoder) retrieval — Task 2's second stage-1 method.

Encodes the whole corpus once into an L2-normalised embedding matrix, then scores a
query by encoding it once and taking a matrix-vector dot product. Because the two
texts are encoded independently, cosine similarity on normalised vectors reduces to a
single dot product — this is exactly the property that makes bi-encoder retrieval
scale to millions of documents (see docs/ARCHITECTURE.md §5, Figure 2), and is the
architectural contrast the cross-encoder in Task 3 is built to explain.

Uses the ``text_raw`` view (see src/data_prep.py) — a transformer encoder was
pretrained on natural text, so lowercasing/stemming it before encoding would only
discard signal the model was trained to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.config import load_config, resolve_device


@dataclass
class ScoredDoc:
    doc_id: str
    score: float


class DenseRetriever:
    """Sentence-embedding bi-encoder retrieval over a fixed corpus."""

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        cfg = load_config()
        self.model_name = model_name or cfg.get("retrieval.dense.model")
        self.device = device or resolve_device(cfg.get("rerank.device", "auto"))
        self.batch_size = batch_size

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.doc_ids: list[str] = []
        self.doc_embeddings: np.ndarray | None = None

    def index(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "DenseRetriever":
        """Encode and store the corpus. One-time cost; reused for every query."""
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must be the same length")

        self.doc_ids = list(doc_ids)
        self.doc_embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return self

    def score_all(self, query_text: str) -> np.ndarray:
        """Cosine similarity of the query against every indexed document.

        Both sides are L2-normalised, so this is a single dot product — the
        precomputability that makes stage-1 dense retrieval cheap.
        """
        if self.doc_embeddings is None:
            raise RuntimeError("call .index() before scoring")
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].astype(np.float32)
        return self.doc_embeddings @ query_vec

    def search(self, query_text: str, top_k: int = 20) -> list[ScoredDoc]:
        scores = self.score_all(query_text)
        top_idx = np.argsort(-scores)[:top_k]
        return [ScoredDoc(self.doc_ids[i], float(scores[i])) for i in top_idx]
