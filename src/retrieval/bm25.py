"""BM25 implemented from the Robertson & Zaragoza formula — not a library call.

General Instruction 9 forbids opaque use of built-in APIs. ``rank_bm25`` is used
**only** as a parity oracle in ``tests/test_bm25.py``; the scoring here is our own.

Formula (Okapi BM25, ``rank_bm25``'s ``BM25Okapi`` convention)::

    score(q, d) = sum_over_terms_t_in_q [ idf(t) * f(t, d) * (k1 + 1) ]
                                        / [ f(t, d) + k1 * (1 - b + b * |d| / avgdl) ]

    idf(t) = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )     # BM25+ / rank_bm25 form

where ``f(t, d)`` is the term frequency of ``t`` in document ``d``, ``|d|`` is the
document length in tokens, ``avgdl`` the corpus average, ``N`` the corpus size, and
``n(t)`` the number of documents containing ``t``.

Input is the ``text_norm`` view (lowercased, tokenized) produced by
``src/data_prep.py`` — BM25 matches surface term forms, so it wants the normalized
view; the cross-encoder in ``src/rerank/cross_encoder.py`` deliberately uses the
opposite (``text_raw``) view. See ``docs/ARCHITECTURE.md`` §4.1 for the justification.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple, deterministic tokenizer for the BM25 (``text_norm``) view.

    Lowercase alphanumeric runs. This is intentionally crude relative to what a
    production system would use (no stemming, no stopword removal) so that the
    BM25-vs-dense comparison in Task 2 isn't confounded by a second, undocumented
    preprocessing difference.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredDoc:
    doc_id: str
    score: float


class BM25:
    """Okapi BM25 over a fixed corpus, scored from the formula above."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon  # floor applied to idf, matching rank_bm25's BM25Okapi

        self.doc_ids: list[str] = []
        self.doc_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self._doc_index: dict[str, int] = {}

    def index(self, doc_ids: Sequence[str], tokenized_docs: Sequence[list[str]]) -> "BM25":
        """Build the index from pre-tokenized documents (call :func:`tokenize` first)."""
        if len(doc_ids) != len(tokenized_docs):
            raise ValueError("doc_ids and tokenized_docs must be the same length")

        self.doc_ids = list(doc_ids)
        self._doc_index = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.doc_freqs = [Counter(tokens) for tokens in tokenized_docs]
        self.doc_lens = [len(tokens) for tokens in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0

        n_docs = len(self.doc_ids)
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] += 1

        # rank_bm25's BM25Okapi idf: log(N - n + 0.5) - log(n + 0.5), floored at
        # eps * average_idf for terms so common the raw formula would go negative.
        raw_idf = {
            term: math.log(n_docs - freq + 0.5) - math.log(freq + 0.5)
            for term, freq in df.items()
        }
        avg_idf = sum(raw_idf.values()) / len(raw_idf) if raw_idf else 0.0
        eps_floor = self.epsilon * avg_idf
        self.idf = {
            term: (score if score > 0 else eps_floor) for term, score in raw_idf.items()
        }
        return self

    def score(self, query_tokens: Iterable[str], doc_index: int) -> float:
        """BM25 score of one document against a tokenized query."""
        freqs = self.doc_freqs[doc_index]
        doc_len = self.doc_lens[doc_index]
        norm = 1 - self.b + self.b * (doc_len / self.avgdl if self.avgdl else 0.0)

        total = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            f = freqs[term]
            idf = self.idf.get(term, 0.0)
            total += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return total

    def score_all(self, query_tokens: Iterable[str]) -> list[float]:
        query_tokens = list(query_tokens)
        return [self.score(query_tokens, i) for i in range(len(self.doc_ids))]

    def search(self, query_text: str, top_k: int = 20) -> list[ScoredDoc]:
        """Tokenize, score against every document, return the top-k by score."""
        query_tokens = tokenize(query_text)
        scores = self.score_all(query_tokens)
        ranked = sorted(
            zip(self.doc_ids, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        return [ScoredDoc(doc_id, score) for doc_id, score in ranked]
