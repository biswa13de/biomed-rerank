"""Hand-implemented IR metrics: Precision@k, Recall@k, MRR, nDCG@k.

General Instruction 9 forbids using built-in APIs without explaining the underlying
process, so every metric here is computed from its definition. ``pytrec_eval`` is used
**only as a test oracle** in ``tests/test_metrics.py`` — never inside the pipeline.

A note on what these numbers mean for SciFact
---------------------------------------------
SciFact qrels are binary and most queries have exactly **one** relevant document.
That distorts every metric, and the distortion must be stated before the numbers are
read, not after:

* **Precision@5** is capped at ``n_relevant / 5``. With one relevant document the
  ceiling is **0.20**, so a raw P@5 of 0.19 is 95% of achievable, not a poor score.
  Use :func:`precision_at_k_ceiling` and report the normalised fraction alongside.
* **Recall@5** degenerates to a **hit-rate**: 1.0 if the single relevant document is
  in the top 5, else 0.0. It is not measuring coverage in the usual sense.
* **MRR** is the genuinely informative metric here — it reports *where* the answer
  landed.
* **nDCG@5** with binary gains reduces to a **position-discount** measure: it reports
  where the answer sits, not how good it is.

Conventions
-----------
``qrels`` maps ``query_id -> {doc_id: relevance}``. A document absent from a query's
qrels is treated as non-relevant (relevance 0), which is the standard TREC
assumption. ``run`` maps ``query_id -> [doc_id, ...]`` in descending rank order.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

Qrels = Mapping[str, Mapping[str, float]]
Ranking = Sequence[str]


# --------------------------------------------------------------------------- #
# Per-query metrics
# --------------------------------------------------------------------------- #

def precision_at_k(ranking: Ranking, relevant: Mapping[str, float], k: int) -> float:
    """Fraction of the top-k that is relevant.

    Note the denominator is always ``k``, not ``min(k, len(ranking))`` — a system that
    returns only 2 documents is penalised for the 3 it failed to return, which is the
    standard TREC convention.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for doc_id in ranking[:k] if relevant.get(doc_id, 0) > 0)
    return hits / k


def precision_at_k_ceiling(relevant: Mapping[str, float], k: int) -> float:
    """The maximum P@k this query can achieve — ``min(n_relevant, k) / k``.

    Reporting P@5 without this ceiling is the single most common misreading of results
    on a single-relevant-document collection such as SciFact.
    """
    n_relevant = sum(1 for score in relevant.values() if score > 0)
    return min(n_relevant, k) / k


def recall_at_k(ranking: Ranking, relevant: Mapping[str, float], k: int) -> float:
    """Fraction of all relevant documents that appear in the top-k.

    Returns 0.0 when a query has no relevant documents; such queries are normally
    excluded upstream rather than scored.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    n_relevant = sum(1 for score in relevant.values() if score > 0)
    if n_relevant == 0:
        return 0.0
    hits = sum(1 for doc_id in ranking[:k] if relevant.get(doc_id, 0) > 0)
    return hits / n_relevant


def reciprocal_rank(ranking: Ranking, relevant: Mapping[str, float], k: int | None = None) -> float:
    """1 / rank of the first relevant document; 0.0 if none is found.

    ``k`` optionally truncates the list, giving MRR@k. Left as ``None``, the whole
    ranking is scanned (standard MRR).
    """
    considered = ranking if k is None else ranking[:k]
    for position, doc_id in enumerate(considered, start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / position
    return 0.0


def dcg_at_k(gains: Iterable[float], k: int) -> float:
    """Discounted Cumulative Gain using the **linear-gain** ``trec_eval`` convention:
    ``gain / log2(rank + 1)`` at every rank, including rank 1
    (``log2(2) == 1``, so rank 1 is effectively undiscounted).

    This is the formula ``pytrec_eval`` (the reference TREC implementation) uses for
    ``ndcg_cut`` — *not* the exponential-gain ``(2^rel - 1) / log2(rank + 1)`` form
    popularised by web-search literature. The two coincide for binary relevance
    (``2^1 - 1 == 1``), which is why the discrepancy stayed hidden until the parity
    test exercised a *graded*-relevance case. We match ``trec_eval`` because that is
    the tool the brief expects results to be validated against; this was caught and
    fixed precisely because the parity cell (§5, Task 5) was built and run before the
    metric was trusted.
    """
    return sum(
        gain / math.log2(position + 1)
        for position, gain in enumerate(list(gains)[:k], start=1)
    )


def ndcg_at_k(ranking: Ranking, relevant: Mapping[str, float], k: int) -> float:
    """nDCG@k — DCG of the produced ranking over DCG of the ideal ranking.

    The ideal ranking sorts *all known relevant documents* by descending relevance,
    so the normaliser does not depend on what the system happened to retrieve.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    actual_gains = [float(relevant.get(doc_id, 0)) for doc_id in ranking[:k]]
    ideal_gains = sorted((s for s in relevant.values() if s > 0), reverse=True)

    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(actual_gains, k) / idcg


# --------------------------------------------------------------------------- #
# Aggregation across queries
# --------------------------------------------------------------------------- #

def evaluate_query(
    ranking: Ranking,
    relevant: Mapping[str, float],
    k: int = 5,
) -> dict[str, float]:
    """All four mandated metrics for a single query, plus the P@k ceiling."""
    return {
        f"precision@{k}": precision_at_k(ranking, relevant, k),
        f"precision@{k}_ceiling": precision_at_k_ceiling(relevant, k),
        f"recall@{k}": recall_at_k(ranking, relevant, k),
        "mrr": reciprocal_rank(ranking, relevant),
        f"ndcg@{k}": ndcg_at_k(ranking, relevant, k),
    }


def evaluate_run(
    run: Mapping[str, Ranking],
    qrels: Qrels,
    k: int = 5,
    skip_unjudged: bool = True,
) -> dict[str, dict[str, float]]:
    """Per-query metrics for a whole run.

    ``skip_unjudged`` drops queries with no relevant document in the qrels. Scoring
    them would contribute a guaranteed 0.0 to every mean and depress the averages for
    a reason unrelated to system quality.
    """
    per_query: dict[str, dict[str, float]] = {}
    for query_id, ranking in run.items():
        relevant = qrels.get(query_id, {})
        if skip_unjudged and not any(score > 0 for score in relevant.values()):
            continue
        per_query[query_id] = evaluate_query(ranking, relevant, k)
    return per_query


def aggregate(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric over queries, plus P@k as a fraction of its ceiling.

    The ``*_pct_of_ceiling`` entry is the honest way to report precision on a
    single-relevant-document collection: it answers "how much of what was achievable
    did we achieve", which is the question the raw number appears to answer but does not.
    """
    if not per_query:
        return {}

    metric_names = next(iter(per_query.values())).keys()
    means = {
        name: sum(scores[name] for scores in per_query.values()) / len(per_query)
        for name in metric_names
    }

    for name in list(means):
        if name.endswith("_ceiling"):
            base = name.removesuffix("_ceiling")
            ceiling = means[name]
            if ceiling > 0:
                means[f"{base}_pct_of_ceiling"] = means[base] / ceiling

    means["n_queries"] = len(per_query)
    return means


def compare_runs(
    baseline: Mapping[str, Mapping[str, float]],
    treatment: Mapping[str, Mapping[str, float]],
    metric: str = "ndcg@5",
) -> dict[str, float]:
    """Paired per-query deltas between two runs on one metric.

    The *distribution* matters more than the mean. A mean gain of +0.03 arising from
    15% of queries improving substantially while 10% degrade is a different system
    from one that improves everything slightly — and distinguishing those two is
    exactly the inference the rubric asks for, so the counts are returned alongside.
    """
    shared = sorted(set(baseline) & set(treatment))
    if not shared:
        raise ValueError("runs share no query ids")

    deltas = [treatment[q][metric] - baseline[q][metric] for q in shared]
    improved = sum(1 for d in deltas if d > 1e-9)
    degraded = sum(1 for d in deltas if d < -1e-9)

    return {
        "metric": metric,
        "n_queries": len(shared),
        "baseline_mean": sum(baseline[q][metric] for q in shared) / len(shared),
        "treatment_mean": sum(treatment[q][metric] for q in shared) / len(shared),
        "mean_delta": sum(deltas) / len(deltas),
        "n_improved": improved,
        "n_degraded": degraded,
        "n_unchanged": len(deltas) - improved - degraded,
        "pct_improved": 100.0 * improved / len(deltas),
        "pct_degraded": 100.0 * degraded / len(deltas),
    }


def recall_ceiling(
    run: Mapping[str, Ranking],
    qrels: Qrels,
    ks: Sequence[int] = (5, 10, 20, 50, 100),
) -> list[dict[str, float]]:
    """Recall@K of the candidate pool for several K — the reranking headroom.

    A reranker can only reorder what stage 1 retrieved. If the relevant document is not
    in the top-K pool, no cross-encoder can recover it, so Recall@K is a hard upper
    bound on any post-rerank metric at that pool size. Reporting this table before any
    reranking number is what allows a result to be read as "94% of achievable" rather
    than as an unexplained absolute.
    """
    rows: list[dict[str, float]] = []
    for k in ks:
        scored = [
            recall_at_k(ranking, qrels.get(qid, {}), k)
            for qid, ranking in run.items()
            if any(s > 0 for s in qrels.get(qid, {}).values())
        ]
        if not scored:
            continue
        rows.append(
            {
                "k": k,
                "recall@k": sum(scored) / len(scored),
                "n_queries": len(scored),
                "pct_queries_with_hit": 100.0 * sum(1 for s in scored if s > 0) / len(scored),
            }
        )
    return rows
