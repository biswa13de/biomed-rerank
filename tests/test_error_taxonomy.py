"""Unit tests for Task 7's automated failure-cause detectors."""

from __future__ import annotations

import pandas as pd

from src.analysis.error_taxonomy import classify_failure


def _corpus_lookup(rows: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame({"doc_id": list(rows), "text_raw": list(rows.values())})
    return df.set_index("doc_id")


def test_flags_truncation_when_relevant_doc_was_truncated():
    delta_rows = pd.DataFrame(
        {
            "query_id": ["q1"],
            "doc_id": ["d1"],
            "rank_stage1": [1],
            "rank_ce": [10],
            "is_relevant": [True],
            "was_truncated": [True],
        }
    )
    corpus_lookup = _corpus_lookup({"d1": "some long document text"})

    result = classify_failure("q1", "a plain query with no abbreviations", delta_rows, corpus_lookup)

    assert "long_document_truncation" in result["causes"]


def test_flags_abbreviation_in_query():
    delta_rows = pd.DataFrame(
        {
            "query_id": ["q1"],
            "doc_id": ["d1"],
            "rank_stage1": [1],
            "rank_ce": [1],
            "is_relevant": [True],
            "was_truncated": [False],
        }
    )
    corpus_lookup = _corpus_lookup({"d1": "irrelevant text"})

    result = classify_failure("q1", "TCR signaling requires CD8 engagement.", delta_rows, corpus_lookup)

    assert "abbreviation_ambiguity" in result["causes"]


def test_flags_negation_near_query_term_in_top_wrong_doc():
    delta_rows = pd.DataFrame(
        {
            "query_id": ["q1"],
            "doc_id": ["wrong1"],
            "rank_stage1": [5],
            "rank_ce": [1],
            "is_relevant": [False],
            "was_truncated": [False],
        }
    )
    corpus_lookup = _corpus_lookup(
        {"wrong1": "This study found no effect of aspirin on platelet aggregation in healthy volunteers."}
    )

    result = classify_failure("q1", "Aspirin reduces platelet aggregation.", delta_rows, corpus_lookup)

    assert "negation_context_misreading" in result["causes"]


def test_falls_back_to_manual_when_no_automated_cause_fires():
    delta_rows = pd.DataFrame(
        {
            "query_id": ["q1"],
            "doc_id": ["d1"],
            "rank_stage1": [1],
            "rank_ce": [1],
            "is_relevant": [True],
            "was_truncated": [False],
        }
    )
    corpus_lookup = _corpus_lookup({"d1": "a short clean document"})

    result = classify_failure("q1", "a plain query", delta_rows, corpus_lookup)

    assert "requires manual annotation" in result["causes"]
    assert result["n_automated_causes"] == 0
