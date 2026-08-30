"""Hybrid retrieval via Reciprocal Rank Fusion (RRF) — Task 2's third stage-1 method.

RRF combines two ranked lists using **rank position only**, not raw scores:

    rrf_score(d) = sum over rankers r that retrieved d of  1 / (k + rank_r(d))

with k = 60 (Cormack et al., 2009). Rank fusion sidesteps the score-calibration
problem that a naive weighted-sum fusion would hit: BM25 scores are unbounded and
corpus-dependent, cosine similarities are bounded in [-1, 1] and cluster near 0 for
unrelated pairs — the two are not on a comparable scale, so averaging them directly
would let whichever method happens to produce larger numbers dominate. Rank position
is comparable by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.retrieval.bm25 import ScoredDoc as BM25ScoredDoc
from src.retrieval.dense import ScoredDoc as DenseScoredDoc


@dataclass
class ScoredDoc:
    doc_id: str
    score: float


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[BM25ScoredDoc | DenseScoredDoc]],
    k: int = 60,
    top_k: int = 20,
) -> list[ScoredDoc]:
    """Fuse any number of pre-ranked result lists by reciprocal rank.

    ``rankings`` is a list of already-sorted (descending score) result lists, e.g.
    ``[bm25.search(q, top_k=100), dense.search(q, top_k=100)]``. A document that only
    one ranker retrieved still gets a (smaller) contribution from that ranker alone —
    it is not required to appear in every input list.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, scored_doc in enumerate(ranking, start=1):
            fused[scored_doc.doc_id] = fused.get(scored_doc.doc_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [ScoredDoc(doc_id, score) for doc_id, score in ranked]


class HybridRetriever:
    """Convenience wrapper fusing a BM25 and a dense retriever via RRF.

    Both underlying retrievers must already be indexed over the same corpus. Each is
    queried for a wider candidate pool (``fusion_pool``) than the final ``top_k``,
    since RRF needs enough overlap between the two rankings to produce a meaningfully
    fused order — fusing two top-20 lists with little overlap degenerates toward the
    union of two arbitrary orderings.
    """

    def __init__(self, bm25, dense, rrf_k: int = 60, fusion_pool: int = 100):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.fusion_pool = fusion_pool

    def search(self, query_text: str, top_k: int = 20) -> list[ScoredDoc]:
        bm25_results = self.bm25.search(query_text, top_k=self.fusion_pool)
        dense_results = self.dense.search(query_text, top_k=self.fusion_pool)
        return reciprocal_rank_fusion(
            [bm25_results, dense_results], k=self.rrf_k, top_k=top_k
        )
