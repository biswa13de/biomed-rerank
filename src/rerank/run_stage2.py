"""Task 3 driver — reranks the Stage-1 hybrid candidate pool with a cross-encoder,
evaluates before vs. after, and saves the worked single-query trace the brief asks for.

Reranks the **hybrid** run rather than BM25 or dense alone: Stage 1's own analysis
(``results/tables/stage1_metrics.csv``) shows hybrid has the best recall ceiling of
the three methods, so it gives the cross-encoder the strongest candidate pool to work
with — reranking a weaker pool would conflate "the reranker underperformed" with "the
reranker was fed a worse pool" (see docs/ARCHITECTURE.md §6, the recall-ceiling logic).

Produces:
    results/tables/stage2_latency.csv       reranking latency, batch of `top_k_default`
    results/tables/stage2_metrics.csv       stage-1 hybrid vs. reranked, P@5/R@5/MRR/nDCG@5
    results/tables/stage2_trace.json        one worked single-query trace (input_ids, logit)
    results/logs/run_manifest.json          appended with this run's provenance
    results/runs/rerank_ms_marco.parquet    full reranked lists, used by Tasks 4/5/6/7

Run with:  .venv/bin/python -m src.rerank.run_stage2
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.eval.metrics import aggregate, evaluate_run
from src.rerank.cross_encoder import CrossEncoder
from src.utils.manifest import write_manifest
from src.utils.seeding import set_seed
from src.utils.timing import StageTimer, measure


def load_processed_and_pool(cfg):
    processed_dir = Path(cfg.get("dataset.processed_dir"))
    corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    queries = pd.read_parquet(processed_dir / "queries.parquet")
    qrels_df = pd.read_parquet(processed_dir / "qrels.parquet")

    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_df.itertuples():
        qrels.setdefault(row.query_id, {})[row.doc_id] = float(row.relevance)

    pool_df = pd.read_parquet(Path(cfg.root) / "results" / "runs" / "hybrid.parquet")
    pool: dict[str, list[str]] = {}
    for qid, group in pool_df.sort_values("rank").groupby("query_id"):
        pool[str(qid)] = group["doc_id"].astype(str).tolist()

    return corpus, queries, qrels, pool


def rerank_all_queries(
    reranker: CrossEncoder,
    queries: pd.DataFrame,
    pool: dict[str, list[str]],
    corpus_lookup: pd.DataFrame,
    top_k_pool: int,
    top_k_out: int,
    timer: StageTimer,
):
    """Rerank each query's top-`top_k_pool` hybrid candidates down to `top_k_out`."""
    run: dict[str, list[str]] = {}
    truncation_flags: list[bool] = []

    for row in queries.itertuples():
        qid = row.query_id
        candidate_ids = pool.get(qid, [])[:top_k_pool]
        if not candidate_ids:
            run[qid] = []
            continue

        candidates = [
            (doc_id, corpus_lookup.loc[doc_id, "text_raw"])
            for doc_id in candidate_ids
            if doc_id in corpus_lookup.index
        ]

        with timer.stage("rerank"):
            reranked = reranker.rerank(row.query_raw, candidates, top_k=top_k_out)

        run[qid] = [r.doc_id for r in reranked]
        truncation_flags.extend(r.was_truncated for r in reranked)

    return run, truncation_flags


def worked_trace(reranker: CrossEncoder, query_row, pool: dict[str, list[str]], corpus_lookup, top_k_pool: int):
    """One query walked end-to-end: hybrid top-K with the cross-encoder's manual trace
    for its top candidate — ARCHITECTURE.md requirement 5.3."""
    qid = query_row.query_id
    candidate_ids = pool.get(qid, [])[:top_k_pool]
    top_doc_id = candidate_ids[0]
    top_doc_text = corpus_lookup.loc[top_doc_id, "text_raw"]

    trace = reranker.explain_one(query_row.query_raw, top_doc_id, top_doc_text)
    return {
        "query_id": qid,
        "query": query_row.query_raw,
        "hybrid_top_k_doc_ids": candidate_ids,
        "traced_doc_id": trace.doc_id,
        "n_tokens_before_truncation": trace.n_tokens_before_truncation,
        "was_truncated": trace.was_truncated,
        "logit": trace.logit,
        "input_ids": trace.input_ids,
        "tokens": trace.tokens,
    }


def main():
    set_seed()
    cfg = load_config()
    corpus, queries, qrels, pool = load_processed_and_pool(cfg)
    corpus_lookup = corpus.set_index("doc_id")

    top_k_pool = cfg.get("retrieval.top_k_default", 20)
    metrics_k = cfg.get("evaluation.cutoff", 5)

    print(f"Loading cross-encoder: {cfg.get('rerank.models.primary')} ...")
    reranker = CrossEncoder()

    print(f"Reranking hybrid top-{top_k_pool} pool for {len(queries)} queries...")
    timer = StageTimer()
    rerank_run, truncation_flags = rerank_all_queries(
        reranker, queries, pool, corpus_lookup, top_k_pool, top_k_pool, timer
    )

    stage1_run = {qid: docs[:top_k_pool] for qid, docs in pool.items()}

    results = {}
    for name, run in [("stage1_hybrid", stage1_run), ("stage2_reranked", rerank_run)]:
        per_query = evaluate_run(run, qrels, k=metrics_k)
        results[name] = aggregate(per_query)
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "method"})

    warmup = cfg.get("timing.warmup_runs", 3)
    repeats = cfg.get("timing.timed_repeats", 5)
    sample_row = queries.iloc[0]
    sample_candidates = [
        (doc_id, corpus_lookup.loc[doc_id, "text_raw"])
        for doc_id in pool.get(sample_row.query_id, [])[:top_k_pool]
        if doc_id in corpus_lookup.index
    ]
    _, rerank_timing = measure(
        lambda: reranker.rerank(sample_row.query_raw, sample_candidates, top_k=top_k_pool),
        f"rerank_k{top_k_pool}",
        warmup,
        repeats,
    )
    latency_df = pd.DataFrame([rerank_timing.as_row()])

    trace = worked_trace(reranker, sample_row, pool, corpus_lookup, top_k_pool)

    tables_dir = cfg.path("output.tables_dir")
    metrics_df.to_csv(tables_dir / "stage2_metrics.csv", index=False)
    latency_df.to_csv(tables_dir / "stage2_latency.csv", index=False)

    import json

    (tables_dir / "stage2_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    runs_dir = Path(cfg.root) / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"query_id": qid, "doc_id": doc_id, "rank": i + 1, "stage": "rerank_ms_marco"}
        for qid, doc_ids in rerank_run.items()
        for i, doc_id in enumerate(doc_ids)
    ]
    pd.DataFrame(rows).to_parquet(runs_dir / "rerank_ms_marco.parquet", index=False)

    n_truncated = sum(truncation_flags)
    write_manifest(
        "task3_cross_encoder_reranking",
        cfg.path("output.logs_dir"),
        config=cfg.as_dict(),
        extra={
            "n_queries": len(queries),
            "top_k_pool": top_k_pool,
            "n_pairs_scored": len(truncation_flags),
            "n_pairs_truncated": n_truncated,
            "pct_pairs_truncated": 100.0 * n_truncated / len(truncation_flags) if truncation_flags else 0.0,
        },
    )

    print("\n=== Stage-1 hybrid vs. Stage-2 reranked (k=5) ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Reranking latency (ms), pool size =", top_k_pool, "===")
    print(latency_df.to_string(index=False))
    print(f"\nPairs truncated at {cfg.get('rerank.max_length')} tokens: {n_truncated}/{len(truncation_flags)}")
    print(f"Saved tables to {tables_dir}, run to {runs_dir}")


if __name__ == "__main__":
    main()
