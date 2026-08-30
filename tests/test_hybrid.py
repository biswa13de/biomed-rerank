"""Unit tests for Reciprocal Rank Fusion — hand-computed, since RRF has no standard
library implementation to check parity against; the formula itself is the spec.
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid import ScoredDoc, reciprocal_rank_fusion


def _sd(doc_id: str, score: float) -> ScoredDoc:
    return ScoredDoc(doc_id, score)


def test_rrf_hand_computed_two_lists():
    bm25 = [_sd("a", 9.0), _sd("b", 5.0), _sd("c", 2.0)]
    dense = [_sd("b", 0.9), _sd("d", 0.8), _sd("a", 0.5)]

    fused = reciprocal_rank_fusion([bm25, dense], k=60, top_k=10)
    scores = {sd.doc_id: sd.score for sd in fused}

    # a: rank1 in bm25 (1/61), rank3 in dense (1/63)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    # b: rank2 in bm25 (1/62), rank1 in dense (1/61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    # c: rank3 in bm25 only (1/63)
    assert scores["c"] == pytest.approx(1 / 63)
    # d: rank2 in dense only (1/62)
    assert scores["d"] == pytest.approx(1 / 62)

    # b has the highest fused score (top of one list, second of the other)
    assert fused[0].doc_id == "b"


def test_rrf_respects_top_k():
    bm25 = [_sd(f"d{i}", 100 - i) for i in range(30)]
    dense = [_sd(f"d{i}", 1.0 - i * 0.01) for i in range(30)]
    fused = reciprocal_rank_fusion([bm25, dense], top_k=5)
    assert len(fused) == 5


def test_rrf_doc_only_in_one_list_still_included():
    bm25 = [_sd("only_bm25", 5.0)]
    dense = [_sd("only_dense", 0.9)]
    fused = reciprocal_rank_fusion([bm25, dense], top_k=10)
    ids = {sd.doc_id for sd in fused}
    assert ids == {"only_bm25", "only_dense"}


def test_rrf_single_list_preserves_order():
    ranking = [_sd("x", 3.0), _sd("y", 2.0), _sd("z", 1.0)]
    fused = reciprocal_rank_fusion([ranking], top_k=10)
    assert [sd.doc_id for sd in fused] == ["x", "y", "z"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], top_k=10) == []
