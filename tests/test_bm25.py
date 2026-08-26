"""Parity test: our hand-implemented BM25 vs rank_bm25's BM25Okapi.

Evidence for General Instruction 9 — the scoring formula is implemented from the
Robertson & Zaragoza definition in src/retrieval/bm25.py; rank_bm25 is used here only
to confirm the implementation, never inside the retrieval pipeline itself.

Run with:  .venv/bin/python -m pytest tests/test_bm25.py -v
"""

from __future__ import annotations

import random

import pytest

from src.retrieval.bm25 import BM25, tokenize

rank_bm25 = pytest.importorskip("rank_bm25")
from rank_bm25 import BM25Okapi

TOLERANCE = 1e-6


def _synthetic_corpus(n_docs: int = 60, seed: int = 11) -> tuple[list[str], list[str]]:
    """Small biomedical-flavoured vocabulary so term overlap resembles the real corpus."""
    rng = random.Random(seed)
    vocab = [
        "gene", "protein", "expression", "cancer", "tumor", "cell", "mutation",
        "patient", "trial", "clinical", "treatment", "therapy", "receptor",
        "inhibitor", "pathway", "signaling", "dna", "rna", "sequencing", "biomarker",
        "abbreviation", "mrna", "vaccine", "antibody", "immune", "response", "dose",
        "efficacy", "placebo", "cohort",
    ]
    doc_ids = [f"doc{i}" for i in range(n_docs)]
    docs = [
        " ".join(rng.choices(vocab, k=rng.randint(20, 120))) for _ in range(n_docs)
    ]
    return doc_ids, docs


def test_bm25_matches_rank_bm25_scores():
    doc_ids, docs = _synthetic_corpus()
    tokenized = [tokenize(d) for d in docs]

    ours = BM25(k1=1.2, b=0.75, epsilon=0.25).index(doc_ids, tokenized)
    theirs = BM25Okapi(tokenized, k1=1.2, b=0.75, epsilon=0.25)

    queries = [
        "gene expression in tumor cell",
        "clinical trial patient response",
        "protein receptor inhibitor pathway",
        "unseen_term_not_in_corpus",
    ]

    for query in queries:
        query_tokens = tokenize(query)
        our_scores = ours.score_all(query_tokens)
        their_scores = theirs.get_scores(query_tokens)

        assert len(our_scores) == len(their_scores)
        for i, (mine, ref) in enumerate(zip(our_scores, their_scores)):
            assert mine == pytest.approx(ref, abs=TOLERANCE), (
                f"score mismatch on {doc_ids[i]} for query {query!r}: "
                f"ours={mine} rank_bm25={ref}"
            )


def test_bm25_top_k_ranking_matches():
    """The ranking order, not just raw scores, must agree — that's what retrieval uses."""
    doc_ids, docs = _synthetic_corpus(n_docs=100, seed=99)
    tokenized = [tokenize(d) for d in docs]

    ours = BM25().index(doc_ids, tokenized)
    theirs = BM25Okapi(tokenized)

    query = "cancer treatment therapy dose efficacy"
    query_tokens = tokenize(query)

    our_top = [sd.doc_id for sd in ours.search(query, top_k=10)]
    their_scores = theirs.get_scores(query_tokens)
    their_top = [
        doc_ids[i]
        for i in sorted(range(len(doc_ids)), key=lambda i: their_scores[i], reverse=True)[:10]
    ]

    assert our_top == their_top


def test_tokenizer_lowercases_and_strips_punctuation():
    assert tokenize("BRCA1 mutations (n=42)") == ["brca1", "mutations", "n", "42"]


def test_empty_query_scores_all_zero():
    doc_ids, docs = _synthetic_corpus(n_docs=10)
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25().index(doc_ids, tokenized)
    assert bm25.score_all([]) == [0.0] * len(doc_ids)
