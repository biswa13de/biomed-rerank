"""Task 4 instrumentation — before/after reranking case mining.

The brief asks for five specific before/after cases, one per **distinct** query. Hand
-hunting these is unreliable and unreviewable, so this module computes, for every
(query, doc) pair that appears in either the Stage-1 hybrid pool or the Stage-2
reranked run, its rank in each and the delta between them, then auto-buckets queries
into the five required categories by a fixed, stated rule per docs' implementation
plan (Task 4). The machine finds candidates; a human still has to read and annotate
the winning row from each bucket — this module only narrows "300 queries" down to
"five auto-selected candidates per bucket" so the annotation is tractable.

One case per query id is enforced by construction: each bucket is chosen from a
disjoint slice of query ids (greedily, in the order the buckets are listed), so no
query can supply two of the five required cases.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.eval.metrics import ndcg_at_k

_ABBREV_RE = re.compile(r"\b[A-Z]{2,6}\b")


def build_rank_delta_table(
    stage1_run: dict[str, list[str]],
    stage2_run: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (query_id, doc_id) appearing in either run.

    ``rank_stage1``/``rank_ce`` are ``None`` when a doc fell outside that run's pool
    (this happens only at the pool boundary, since Stage 2 reranks the same candidate
    set Stage 1 produced) — such rows contribute no `delta` and are excluded from
    delta-based bucketing but kept for completeness.
    """
    corpus_lookup = corpus.set_index("doc_id")
    query_lookup = queries.set_index("query_id")
    rows = []

    for qid in set(stage1_run) | set(stage2_run):
        s1 = stage1_run.get(qid, [])
        s2 = stage2_run.get(qid, [])
        relevant = qrels.get(qid, {})
        doc_ids = set(s1) | set(s2)

        for doc_id in doc_ids:
            rank_stage1 = s1.index(doc_id) + 1 if doc_id in s1 else None
            rank_ce = s2.index(doc_id) + 1 if doc_id in s2 else None
            delta = (rank_stage1 - rank_ce) if (rank_stage1 and rank_ce) else None

            rows.append(
                {
                    "query_id": qid,
                    "query": query_lookup.loc[qid, "query_raw"] if qid in query_lookup.index else "",
                    "doc_id": doc_id,
                    "doc_title": corpus_lookup.loc[doc_id, "title"] if doc_id in corpus_lookup.index else "",
                    "rank_stage1": rank_stage1,
                    "rank_ce": rank_ce,
                    "delta": delta,
                    "is_relevant": relevant.get(doc_id, 0) > 0,
                    "was_truncated": corpus_lookup.loc[doc_id, "n_tokens"] > 512
                    if doc_id in corpus_lookup.index
                    else False,
                }
            )

    return pd.DataFrame(rows)


def per_query_ndcg_delta(
    stage1_run: dict[str, list[str]],
    stage2_run: dict[str, list[str]],
    qrels: dict[str, dict[str, float]],
    k: int = 5,
) -> pd.DataFrame:
    """nDCG@k for stage1 vs. stage2 reranked, per query — used by the "failure" bucket."""
    rows = []
    for qid in set(stage1_run) & set(stage2_run):
        relevant = qrels.get(qid, {})
        if not any(v > 0 for v in relevant.values()):
            continue
        ndcg_stage1 = ndcg_at_k(stage1_run[qid], relevant, k)
        ndcg_ce = ndcg_at_k(stage2_run[qid], relevant, k)
        rows.append(
            {
                "query_id": qid,
                f"ndcg@{k}_stage1": ndcg_stage1,
                f"ndcg@{k}_ce": ndcg_ce,
                "ndcg_delta": ndcg_ce - ndcg_stage1,
            }
        )
    return pd.DataFrame(rows)


def select_task4_cases(
    delta_df: pd.DataFrame,
    ndcg_df: pd.DataFrame,
    queries: pd.DataFrame,
) -> pd.DataFrame:
    """Auto-select one candidate query per required Task 4 case, per the fixed rules
    in the implementation plan. Query ids are consumed greedily and removed from the
    pool as each bucket claims one, guaranteeing five *distinct* queries.
    """
    remaining_qids = set(delta_df["query_id"].unique())
    selected: list[dict] = []

    def claim(qid: str, case: str, note: str):
        remaining_qids.discard(qid)
        selected.append({"case": case, "query_id": qid, "note": note})

    # 1. Relevant promoted: is_relevant, rank_stage1 > 5, rank_ce <= 3.
    promoted = delta_df[
        delta_df["query_id"].isin(remaining_qids)
        & delta_df["is_relevant"]
        & (delta_df["rank_stage1"] > 5)
        & (delta_df["rank_ce"] <= 3)
    ].sort_values("delta", ascending=False)
    if len(promoted):
        row = promoted.iloc[0]
        claim(row["query_id"], "relevant_promoted", f"doc {row['doc_id']} moved rank {row['rank_stage1']}->{row['rank_ce']}")

    # 2. Irrelevant demoted: not is_relevant, rank_stage1 <= 3, rank_ce > 5.
    demoted = delta_df[
        delta_df["query_id"].isin(remaining_qids)
        & (~delta_df["is_relevant"])
        & (delta_df["rank_stage1"] <= 3)
        & (delta_df["rank_ce"] > 5)
    ].sort_values("delta")
    if len(demoted):
        row = demoted.iloc[0]
        claim(row["query_id"], "irrelevant_demoted", f"doc {row['doc_id']} moved rank {row['rank_stage1']}->{row['rank_ce']}")

    # 3. Minimal change: max |delta| over top-5 (either run) <= 1.
    top5 = delta_df[(delta_df["rank_stage1"] <= 5) | (delta_df["rank_ce"] <= 5)]
    minimal_candidates = (
        top5[top5["query_id"].isin(remaining_qids) & top5["delta"].notna()]
        .groupby("query_id")["delta"]
        .apply(lambda s: s.abs().max())
    )
    minimal_qids = minimal_candidates[minimal_candidates <= 1].sort_values()
    if len(minimal_qids):
        qid = minimal_qids.index[0]
        claim(qid, "minimal_change", f"max |delta| in top-5 = {minimal_qids.iloc[0]}")

    # 4. Failure: per-query nDCG@5_ce < nDCG@5_stage1, worst regression first.
    failures = ndcg_df[
        ndcg_df["query_id"].isin(remaining_qids) & (ndcg_df["ndcg_delta"] < 0)
    ].sort_values("ndcg_delta")
    if len(failures):
        row = failures.iloc[0]
        claim(row["query_id"], "failure", f"nDCG@5 dropped by {abs(row['ndcg_delta']):.3f}")

    # 5. Terminology/abbreviation: query matches abbreviation regex, sorted by |delta|.
    query_lookup = queries.set_index("query_id")
    abbrev_qids = {
        qid for qid in remaining_qids
        if qid in query_lookup.index and _ABBREV_RE.search(query_lookup.loc[qid, "query_raw"])
    }
    abbrev_deltas = (
        delta_df[delta_df["query_id"].isin(abbrev_qids) & delta_df["delta"].notna()]
        .groupby("query_id")["delta"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=False)
    )
    if len(abbrev_deltas):
        qid = abbrev_deltas.index[0]
        claim(qid, "terminology_abbreviation", f"query contains abbreviation, max |delta| = {abbrev_deltas.iloc[0]}")

    result = pd.DataFrame(selected)
    if not result.empty:
        result = result.merge(query_lookup[["query_raw"]], left_on="query_id", right_index=True, how="left")
    return result


def load_runs_and_data(cfg):
    processed_dir = Path(cfg.get("dataset.processed_dir"))
    corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    queries = pd.read_parquet(processed_dir / "queries.parquet")
    qrels_df = pd.read_parquet(processed_dir / "qrels.parquet")

    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_df.itertuples():
        qrels.setdefault(row.query_id, {})[row.doc_id] = float(row.relevance)

    runs_dir = Path(cfg.root) / "results" / "runs"

    def load_run(name: str) -> dict[str, list[str]]:
        df = pd.read_parquet(runs_dir / f"{name}.parquet")
        run: dict[str, list[str]] = {}
        for qid, group in df.sort_values("rank").groupby("query_id"):
            run[str(qid)] = group["doc_id"].astype(str).tolist()
        return run

    stage1_run = load_run("hybrid")
    stage2_run = load_run("rerank_ms_marco")
    return corpus, queries, qrels, stage1_run, stage2_run


def main():
    cfg = load_config()
    corpus, queries, qrels, stage1_run, stage2_run = load_runs_and_data(cfg)

    top_k_pool = cfg.get("retrieval.top_k_default", 20)
    stage1_run = {qid: docs[:top_k_pool] for qid, docs in stage1_run.items()}

    delta_df = build_rank_delta_table(stage1_run, stage2_run, qrels, queries, corpus)
    ndcg_df = per_query_ndcg_delta(stage1_run, stage2_run, qrels, k=cfg.get("evaluation.cutoff", 5))
    cases_df = select_task4_cases(delta_df, ndcg_df, queries)

    tables_dir = cfg.path("output.tables_dir")
    delta_df.to_csv(tables_dir / "task4_rank_deltas.csv", index=False)
    ndcg_df.to_csv(tables_dir / "task4_ndcg_deltas.csv", index=False)
    cases_df.to_csv(tables_dir / "task4_selected_cases.csv", index=False)

    print(f"Rank-delta rows: {len(delta_df)}")
    print(f"Per-query nDCG deltas: {len(ndcg_df)}")
    print("\n=== Auto-selected Task 4 cases (one per distinct query) ===")
    print(cases_df.to_string(index=False))
    print(f"\nSaved tables to {tables_dir}")


if __name__ == "__main__":
    main()
