"""Unit tests for the manually-tokenized cross-encoder reranker.

Loads the real (small, already-cached) primary model rather than mocking — the point
of Task 3 is the manual tokenization/logit-extraction path itself, and a mock would
test nothing about whether that path is wired correctly.
"""

from __future__ import annotations

import pytest

from src.rerank.cross_encoder import CrossEncoder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cross_encoder():
    return CrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=4, max_length=512)


def test_rerank_sorts_by_score_descending(cross_encoder):
    query = "What is the capital of France?"
    candidates = [
        ("relevant", "Paris is the capital and largest city of France."),
        ("irrelevant", "Photosynthesis converts sunlight into chemical energy in plants."),
    ]
    results = cross_encoder.rerank(query, candidates)

    assert [r.doc_id for r in results][0] == "relevant"
    assert results[0].score > results[1].score


def test_rerank_respects_top_k(cross_encoder):
    query = "test query"
    candidates = [(f"doc{i}", f"some document text number {i}") for i in range(10)]
    results = cross_encoder.rerank(query, candidates, top_k=3)
    assert len(results) == 3


def test_truncation_flagged_for_long_document(cross_encoder):
    query = "short query"
    long_doc = "word " * 2000
    short_doc = "a short document"
    results = cross_encoder.rerank(query, [("long", long_doc), ("short", short_doc)])

    by_id = {r.doc_id: r for r in results}
    assert by_id["long"].was_truncated is True
    assert by_id["long"].n_tokens_before_truncation > 512
    assert by_id["short"].was_truncated is False


def test_explain_one_exposes_input_ids_and_logit(cross_encoder):
    trace = cross_encoder.explain_one("capital of France", "doc1", "Paris is the capital of France.")

    assert trace.input_ids[0] == cross_encoder.tokenizer.cls_token_id
    assert cross_encoder.tokenizer.sep_token_id in trace.input_ids
    assert isinstance(trace.logit, float)
    assert len(trace.tokens) == len(trace.input_ids)
    assert trace.was_truncated is False
