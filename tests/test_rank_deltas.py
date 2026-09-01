"""Unit tests for Task 4's automated before/after case-mining logic."""

from __future__ import annotations

import pandas as pd

from src.analysis.rank_deltas import build_rank_delta_table, select_task4_cases


def _queries_df(query_ids):
    return pd.DataFrame({"query_id": query_ids, "query_raw": [f"query {q}" for q in query_ids]})


def _corpus_df(doc_ids):
    return pd.DataFrame(
        {
            "doc_id": doc_ids,
            "title": [f"title {d}" for d in doc_ids],
            "n_tokens": [100] * len(doc_ids),
        }
    )


def test_build_rank_delta_table_computes_deltas():
    stage1 = {"q1": ["a", "b", "c"]}
    stage2 = {"q1": ["c", "a", "b"]}
    qrels = {"q1": {"c": 1}}
    queries = _queries_df(["q1"])
    corpus = _corpus_df(["a", "b", "c"])

    df = build_rank_delta_table(stage1, stage2, qrels, queries, corpus)
    row_c = df[df["doc_id"] == "c"].iloc[0]

    assert row_c["rank_stage1"] == 3
    assert row_c["rank_ce"] == 1
    assert row_c["delta"] == 2
    assert row_c["is_relevant"] == True


def test_select_task4_cases_assigns_distinct_queries():
    # q1: relevant doc promoted from rank 10 to rank 1.
    stage1 = {
        "q1": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "rel1"],
        "q2": ["irr1", "irr2", "y3", "y4", "y5", "y6"],
    }
    stage2 = {
        "q1": ["rel1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"],
        "q2": ["y3", "y4", "y5", "y6", "irr1", "irr2"],
    }
    qrels = {"q1": {"rel1": 1}, "q2": {"y3": 1}}
    queries = _queries_df(["q1", "q2"])
    corpus = _corpus_df(["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "rel1", "irr1", "irr2", "y3", "y4", "y5", "y6"])

    delta_df = build_rank_delta_table(stage1, stage2, qrels, queries, corpus)
    ndcg_df = pd.DataFrame(columns=["query_id", "ndcg_delta"])

    cases = select_task4_cases(delta_df, ndcg_df, queries)

    assert cases[cases["case"] == "relevant_promoted"]["query_id"].iloc[0] == "q1"
    assert cases[cases["case"] == "irrelevant_demoted"]["query_id"].iloc[0] == "q2"
    # each claimed query id appears in at most one case
    assert cases["query_id"].is_unique
