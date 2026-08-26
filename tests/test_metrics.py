"""Parity tests: our hand-implemented metrics vs pytrec_eval (the TREC reference).

This is the evidence for General Instruction 9. We implement the metrics ourselves and
then prove the implementation correct against the standard tool, rather than calling
the standard tool and asserting it works. ``pytrec_eval`` appears here and nowhere in
the pipeline.

Run with:  .venv/bin/python -m pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import random

import pytest

from src.eval.metrics import (
    aggregate,
    compare_runs,
    dcg_at_k,
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    precision_at_k_ceiling,
    recall_at_k,
    recall_ceiling,
    reciprocal_rank,
)

pytrec_eval = pytest.importorskip("pytrec_eval")

TOLERANCE = 1e-9


# --------------------------------------------------------------------------- #
# Hand-computed cases — verify the definitions before trusting the oracle
# --------------------------------------------------------------------------- #

def test_precision_at_k_hand_computed():
    ranking = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2": 1, "d5": 1}
    assert precision_at_k(ranking, relevant, 5) == pytest.approx(2 / 5)
    assert precision_at_k(ranking, relevant, 3) == pytest.approx(1 / 3)
    assert precision_at_k(ranking, relevant, 1) == pytest.approx(0.0)


def test_precision_ceiling_is_capped_by_relevant_count():
    """The SciFact case: one relevant doc caps P@5 at 0.20."""
    assert precision_at_k_ceiling({"d1": 1}, 5) == pytest.approx(0.20)
    assert precision_at_k_ceiling({"d1": 1, "d2": 1}, 5) == pytest.approx(0.40)
    # More relevant docs than k: the ceiling saturates at 1.0.
    assert precision_at_k_ceiling({f"d{i}": 1 for i in range(10)}, 5) == pytest.approx(1.0)


def test_recall_degenerates_to_hit_rate_with_one_relevant_doc():
    """With a single relevant document, Recall@5 is binary, not a coverage fraction."""
    relevant = {"d3": 1}
    assert recall_at_k(["d1", "d2", "d3", "d4", "d5"], relevant, 5) == 1.0
    assert recall_at_k(["d1", "d2", "d4", "d5", "d6"], relevant, 5) == 0.0


def test_reciprocal_rank_hand_computed():
    relevant = {"d3": 1}
    assert reciprocal_rank(["d1", "d2", "d3"], relevant) == pytest.approx(1 / 3)
    assert reciprocal_rank(["d3", "d1", "d2"], relevant) == pytest.approx(1.0)
    assert reciprocal_rank(["d1", "d2"], relevant) == pytest.approx(0.0)


def test_ndcg_perfect_ranking_is_one():
    relevant = {"d1": 1, "d2": 1}
    assert ndcg_at_k(["d1", "d2", "d3"], relevant, 5) == pytest.approx(1.0)


def test_ndcg_position_discount():
    """With binary gains nDCG measures *where* the answer sits."""
    relevant = {"d1": 1}
    at_rank_1 = ndcg_at_k(["d1", "x", "y"], relevant, 5)
    at_rank_3 = ndcg_at_k(["x", "y", "d1"], relevant, 5)
    assert at_rank_1 == pytest.approx(1.0)
    assert at_rank_3 < at_rank_1
    assert at_rank_3 == pytest.approx(0.5)  # 1/log2(3+1) = 0.5


def test_dcg_uses_trec_linear_gain_form():
    # gain / log2(rank + 1): rank1 -> 2/log2(2)=2.0, rank2 -> 1/log2(3)
    expected = 2.0 / 1.0 + 1.0 / 1.584962500721156
    assert dcg_at_k([2, 1], 2) == pytest.approx(expected)


def test_empty_and_degenerate_inputs():
    assert recall_at_k(["d1"], {}, 5) == 0.0
    assert ndcg_at_k(["d1"], {}, 5) == 0.0
    assert reciprocal_rank([], {"d1": 1}) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(["d1"], {"d1": 1}, 0)


# --------------------------------------------------------------------------- #
# Parity against pytrec_eval — the headline check
# --------------------------------------------------------------------------- #

def _synthetic_collection(n_queries: int = 40, pool: int = 30, seed: int = 42):
    """Random runs and qrels, mixing single- and multi-relevant queries.

    Deliberately includes graded relevance so the nDCG parity check exercises the
    exponential-gain path rather than only the binary case SciFact happens to use.
    """
    rng = random.Random(seed)
    doc_ids = [f"doc{i}" for i in range(200)]

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, list[str]] = {}

    for q in range(n_queries):
        qid = f"q{q}"
        n_relevant = rng.choice([1, 1, 1, 2, 3])  # SciFact-like skew toward one
        relevant_docs = rng.sample(doc_ids, n_relevant)
        qrels[qid] = {d: rng.choice([1, 1, 2]) for d in relevant_docs}

        retrieved = rng.sample(doc_ids, pool)
        # Plant some relevant docs in the pool so metrics are not all zero.
        for doc in relevant_docs:
            if rng.random() < 0.75 and doc not in retrieved:
                retrieved[rng.randrange(pool)] = doc
        run[qid] = retrieved

    return qrels, run


def _pytrec_scores(qrels, run, k):
    """Reference scores from pytrec_eval, which needs float scores rather than ranks."""
    formatted_run = {
        qid: {doc: float(len(ranking) - i) for i, doc in enumerate(ranking)}
        for qid, ranking in run.items()
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        {q: dict(rel) for q, rel in qrels.items()},
        {f"P_{k}", f"recall_{k}", f"ndcg_cut_{k}", "recip_rank"},
    )
    return evaluator.evaluate(formatted_run)


@pytest.mark.parametrize("k", [5, 10])
def test_parity_with_pytrec_eval(k):
    """Every metric, every query, must match the TREC reference implementation."""
    qrels, run = _synthetic_collection()
    ours = evaluate_run(run, qrels, k=k, skip_unjudged=False)
    theirs = _pytrec_scores(qrels, run, k)

    for qid, our_scores in ours.items():
        ref = theirs[qid]
        assert our_scores[f"precision@{k}"] == pytest.approx(ref[f"P_{k}"], abs=TOLERANCE), (
            f"P@{k} mismatch on {qid}"
        )
        assert our_scores[f"recall@{k}"] == pytest.approx(ref[f"recall_{k}"], abs=TOLERANCE), (
            f"Recall@{k} mismatch on {qid}"
        )
        assert our_scores[f"ndcg@{k}"] == pytest.approx(ref[f"ndcg_cut_{k}"], abs=TOLERANCE), (
            f"nDCG@{k} mismatch on {qid}"
        )
        assert our_scores["mrr"] == pytest.approx(ref["recip_rank"], abs=TOLERANCE), (
            f"MRR mismatch on {qid}"
        )


def test_parity_holds_for_binary_single_relevant_case():
    """The SciFact regime specifically: binary labels, exactly one relevant doc."""
    rng = random.Random(7)
    doc_ids = [f"doc{i}" for i in range(100)]
    qrels = {f"q{i}": {rng.choice(doc_ids): 1} for i in range(50)}
    run = {qid: rng.sample(doc_ids, 20) for qid in qrels}

    ours = evaluate_run(run, qrels, k=5, skip_unjudged=False)
    theirs = _pytrec_scores(qrels, run, 5)

    for qid, our_scores in ours.items():
        assert our_scores["ndcg@5"] == pytest.approx(theirs[qid]["ndcg_cut_5"], abs=TOLERANCE)
        assert our_scores["precision@5"] == pytest.approx(theirs[qid]["P_5"], abs=TOLERANCE)
        # And the ceiling claim holds: P@5 can never exceed 0.20 here.
        assert our_scores["precision@5"] <= 0.20 + TOLERANCE


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

def test_aggregate_reports_pct_of_ceiling():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {"q1": ["d1", "x", "y", "z", "w"], "q2": ["x", "d2", "y", "z", "w"]}
    means = aggregate(evaluate_run(run, qrels, k=5))

    assert means["precision@5"] == pytest.approx(0.20)
    assert means["precision@5_ceiling"] == pytest.approx(0.20)
    # Both queries found their document inside the top 5: 100% of achievable.
    assert means["precision@5_pct_of_ceiling"] == pytest.approx(1.0)
    assert means["n_queries"] == 2


def test_compare_runs_counts_the_delta_distribution():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    before = {"q1": ["x", "d1"], "q2": ["d2", "x"], "q3": ["x", "d3"]}
    after = {"q1": ["d1", "x"], "q2": ["x", "d2"], "q3": ["x", "d3"]}

    result = compare_runs(
        evaluate_run(before, qrels, k=5), evaluate_run(after, qrels, k=5), "ndcg@5"
    )
    assert result["n_improved"] == 1   # q1 promoted
    assert result["n_degraded"] == 1   # q2 demoted
    assert result["n_unchanged"] == 1  # q3 static


def test_recall_ceiling_is_monotonic_in_k():
    """Headroom cannot shrink as the pool grows — a sanity check on the framing."""
    qrels, run = _synthetic_collection(n_queries=30, pool=100)
    rows = recall_ceiling(run, qrels, ks=(5, 10, 20, 50, 100))
    recalls = [row["recall@k"] for row in rows]
    assert recalls == sorted(recalls), "Recall@K must be non-decreasing in K"
    assert rows[-1]["k"] == 100
