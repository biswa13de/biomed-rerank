"""Task 2 driver — runs BM25, dense, and hybrid retrieval over every SciFact test
query, measures latency, evaluates all three, and mines the qualitative examples the
brief asks for (lexical mismatch, semantic drift, relevant-but-lower-ranked).

Produces:
    results/tables/stage1_latency.csv          per-method p50/p95 latency
    results/tables/stage1_metrics.csv          P@5/R@5/MRR/nDCG@5/recall-ceiling per method
    results/tables/stage1_lexical_mismatch.csv candidate cases: BM25 misses a synonym
    results/tables/stage1_semantic_drift.csv   candidate cases: dense pulls a topical-but-irrelevant doc
    results/logs/run_manifest.json             appended with this run's provenance
    runs/{bm25,dense,hybrid}.parquet           full ranked lists, used by later tasks

Run with:  .venv/bin/python -m src.retrieval.run_stage1
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.eval.metrics import aggregate, evaluate_run, recall_ceiling
from src.retrieval.bm25 import BM25, tokenize
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.utils.manifest import write_manifest
from src.utils.seeding import set_seed
from src.utils.timing import StageTimer, measure


def load_processed():
    cfg = load_config()
    processed_dir = Path(cfg.get("dataset.processed_dir"))
    corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    queries = pd.read_parquet(processed_dir / "queries.parquet")
    qrels_df = pd.read_parquet(processed_dir / "qrels.parquet")

    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_df.itertuples():
        qrels.setdefault(row.query_id, {})[row.doc_id] = float(row.relevance)

    return corpus, queries, qrels


def build_retrievers(corpus: pd.DataFrame, cfg):
    bm25 = BM25(
        k1=cfg.get("retrieval.bm25.k1"),
        b=cfg.get("retrieval.bm25.b"),
        epsilon=cfg.get("retrieval.bm25.epsilon"),
    )
    bm25.index(corpus["doc_id"].tolist(), [tokenize(t) for t in corpus["text_norm"]])

    dense = DenseRetriever(
        model_name=cfg.get("retrieval.dense.model"),
        batch_size=cfg.get("retrieval.dense.batch_size"),
    )
    dense.index(corpus["doc_id"].tolist(), corpus["text_raw"].tolist())

    hybrid = HybridRetriever(bm25, dense, rrf_k=cfg.get("retrieval.hybrid.rrf_k"))
    return bm25, dense, hybrid


def run_all_queries(retriever, queries: pd.DataFrame, text_col: str, top_k: int, timer: StageTimer, stage_name: str):
    """Run one retriever over every query, recording per-query latency under `stage_name`."""
    run: dict[str, list[str]] = {}
    for row in queries.itertuples():
        query_text = getattr(row, text_col)
        with timer.stage(stage_name):
            results = retriever.search(query_text, top_k=top_k)
        run[row.query_id] = [sd.doc_id for sd in results]
    return run


def mine_lexical_mismatch(bm25_run, dense_run, qrels, queries_df, corpus_df, top_n=5):
    """Cases where the relevant doc is missing/low from BM25 but present/high in dense —
    a candidate synonym/paraphrase mismatch BM25's surface-term matching cannot bridge."""
    corpus_lookup = corpus_df.set_index("doc_id")
    rows = []
    for row in queries_df.itertuples():
        qid = row.query_id
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        if not relevant:
            continue
        bm25_ranking = bm25_run.get(qid, [])
        dense_ranking = dense_run.get(qid, [])
        for doc_id in relevant:
            bm25_rank = bm25_ranking.index(doc_id) + 1 if doc_id in bm25_ranking else None
            dense_rank = dense_ranking.index(doc_id) + 1 if doc_id in dense_ranking else None
            if dense_rank is not None and dense_rank <= 5 and (bm25_rank is None or bm25_rank > 15):
                rows.append(
                    {
                        "query_id": qid,
                        "query": row.query_raw,
                        "doc_id": doc_id,
                        "doc_title": corpus_lookup.loc[doc_id, "title"] if doc_id in corpus_lookup.index else "",
                        "bm25_rank": bm25_rank or "not in top-100",
                        "dense_rank": dense_rank,
                    }
                )
    return pd.DataFrame(rows).head(top_n)


def mine_semantic_drift(dense_run, qrels, queries_df, corpus_df, top_n=5):
    """Cases where dense retrieval ranks a *non-relevant* document highly — a candidate
    topical-similarity-without-relevance error a bi-encoder is prone to since it can
    only compare independently-produced summaries, never the query and doc jointly."""
    corpus_lookup = corpus_df.set_index("doc_id")
    rows = []
    for row in queries_df.itertuples():
        qid = row.query_id
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        for rank, doc_id in enumerate(dense_run.get(qid, [])[:3], start=1):
            if doc_id not in relevant:
                rows.append(
                    {
                        "query_id": qid,
                        "query": row.query_raw,
                        "doc_id": doc_id,
                        "doc_title": corpus_lookup.loc[doc_id, "title"] if doc_id in corpus_lookup.index else "",
                        "dense_rank": rank,
                    }
                )
    return pd.DataFrame(rows).head(top_n)


def main():
    set_seed()
    cfg = load_config()
    corpus, queries, qrels = load_processed()

    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} | Building indices...")
    bm25, dense, hybrid = build_retrievers(corpus, cfg)

    top_k = cfg.get("retrieval.top_k_default", 20)
    # The recall-ceiling table (docs/ARCHITECTURE.md §6) needs headroom out to the
    # largest K we ever rerank at; retrieving only `top_k_default` and then asking for
    # Recall@100 would silently report Recall@20 relabelled as Recall@100. So every
    # run pulls the max of the two, and downstream reranking/Task 6 experiments slice
    # the prefix they need from this single wider run.
    max_k = max(top_k, *cfg.get("retrieval.recall_ceiling_ks", [top_k]))
    timer = StageTimer()

    print("Running BM25 over all queries...")
    bm25_run = run_all_queries(bm25, queries, "query_norm", max_k, timer, "bm25")

    print("Running dense retrieval over all queries...")
    dense_run = run_all_queries(dense, queries, "query_raw", max_k, timer, "dense")

    print("Running hybrid (RRF) over all queries...")
    hybrid_run = run_all_queries(hybrid, queries, "query_raw", max_k, timer, "hybrid")

    # Latency: measure() with proper warmup for one representative query per method,
    # matching the brief's "3 warmup runs, 5 timed repeats, median and p95" spec.
    # (The full-query-set loop above gives us per-query totals for the analysis;
    # this gives us the clean, warmed-up single-query latency figure for reporting.)
    sample_query = queries.iloc[0]
    warmup = cfg.get("timing.warmup_runs", 3)
    repeats = cfg.get("timing.timed_repeats", 5)
    _, bm25_timing = measure(lambda: bm25.search(sample_query.query_norm, top_k), "bm25", warmup, repeats)
    _, dense_timing = measure(lambda: dense.search(sample_query.query_raw, top_k), "dense", warmup, repeats)
    _, hybrid_timing = measure(lambda: hybrid.search(sample_query.query_raw, top_k), "hybrid", warmup, repeats)

    latency_df = pd.DataFrame([bm25_timing.as_row(), dense_timing.as_row(), hybrid_timing.as_row()])

    metrics_k = cfg.get("evaluation.cutoff", 5)
    results = {}
    for name, run in [("bm25", bm25_run), ("dense", dense_run), ("hybrid", hybrid_run)]:
        per_query = evaluate_run(run, qrels, k=metrics_k)
        results[name] = aggregate(per_query)
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "method"})

    ceiling_rows = []
    for name, run in [("bm25", bm25_run), ("dense", dense_run), ("hybrid", hybrid_run)]:
        for row in recall_ceiling(run, qrels, ks=cfg.get("retrieval.recall_ceiling_ks")):
            row["method"] = name
            ceiling_rows.append(row)
    ceiling_df = pd.DataFrame(ceiling_rows)

    lexical_df = mine_lexical_mismatch(bm25_run, dense_run, qrels, queries, corpus)
    drift_df = mine_semantic_drift(dense_run, qrels, queries, corpus)

    tables_dir = cfg.path("output.tables_dir")
    latency_df.to_csv(tables_dir / "stage1_latency.csv", index=False)
    metrics_df.to_csv(tables_dir / "stage1_metrics.csv", index=False)
    ceiling_df.to_csv(tables_dir / "stage1_recall_ceiling.csv", index=False)
    lexical_df.to_csv(tables_dir / "stage1_lexical_mismatch.csv", index=False)
    drift_df.to_csv(tables_dir / "stage1_semantic_drift.csv", index=False)

    runs_dir = Path(cfg.root / "results" / "runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in [("bm25", bm25_run), ("dense", dense_run), ("hybrid", hybrid_run)]:
        rows = [
            {"query_id": qid, "doc_id": doc_id, "rank": i + 1, "stage": name}
            for qid, doc_ids in run.items()
            for i, doc_id in enumerate(doc_ids)
        ]
        pd.DataFrame(rows).to_parquet(runs_dir / f"{name}.parquet", index=False)

    write_manifest(
        "task2_stage1_retrieval",
        cfg.path("output.logs_dir"),
        config=cfg.as_dict(),
        extra={"n_queries": len(queries), "n_corpus": len(corpus), "top_k": top_k},
    )

    print("\n=== Latency (ms) ===")
    print(latency_df.to_string(index=False))
    print("\n=== Metrics @k=5 ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Recall ceiling ===")
    print(ceiling_df.pivot(index="k", columns="method", values="recall@k").to_string())
    print(f"\nLexical mismatch candidates found: {len(lexical_df)}")
    print(f"Semantic drift candidates found: {len(drift_df)}")
    print(f"\nSaved tables to {tables_dir}, runs to {runs_dir}")


if __name__ == "__main__":
    main()
