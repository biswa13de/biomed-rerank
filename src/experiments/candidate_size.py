"""Task 6 — candidate-pool-size × model grid experiment.

The brief allows either axis (pool size, or model comparison) — "Alternatively" makes
one sufficient. This does both: a 2x4 grid at one extra loop's cost, since it is the
only place in the assignment ms-marco-MiniLM-L-6 (general-domain) and MedCPT
(domain-matched, trained on PubMed query-article pairs) are compared directly, which
is exactly the "how does biomedical terminology affect relevance scoring" story
Task 3 asks for but cannot answer with only one reranker.

Experimental Constraint (brief's explicit clause, reproduced from
docs/ARCHITECTURE.md §7): hardware, batch size, seed, and query set are held fixed
across every cell of the grid. Only pool size K and reranker model vary. The Stage-1
hybrid run is the single fixed candidate source for every cell — an identical top-50
pool is truncated to K for each row, so the K=5 pool for a query is always a prefix of
its K=50 pool, and no cell sees a differently-ordered upstream retrieval.

Produces:
    results/tables/stage6_grid.csv        one row per (model, K): nDCG@5, latency, n_queries
    results/figures/stage6_ndcg_vs_k.png  nDCG@5 vs K, one line per model (saturation)
    results/figures/stage6_latency_vs_k.png  latency vs K, one line per model (linear)
    results/logs/run_manifest.json        appended with this run's provenance

Run with:  .venv/bin/python -m src.experiments.candidate_size
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


def rerank_at_k(
    reranker: CrossEncoder,
    queries: pd.DataFrame,
    pool: dict[str, list[str]],
    corpus_lookup: pd.DataFrame,
    k: int,
) -> dict[str, list[str]]:
    run: dict[str, list[str]] = {}
    for row in queries.itertuples():
        qid = row.query_id
        candidate_ids = pool.get(qid, [])[:k]
        candidates = [
            (doc_id, corpus_lookup.loc[doc_id, "text_raw"])
            for doc_id in candidate_ids
            if doc_id in corpus_lookup.index
        ]
        if not candidates:
            run[qid] = []
            continue
        reranked = reranker.rerank(row.query_raw, candidates, top_k=k)
        run[qid] = [r.doc_id for r in reranked]
    return run


def resolve_comparison_model(cfg) -> str:
    """MedCPT if it loads, else the declared fallback — checked once, up front, so a
    sandbox without internet access degrades to the fallback instead of failing the
    whole grid partway through."""
    primary_choice = cfg.get("rerank.models.comparison")
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(primary_choice)
        return primary_choice
    except Exception as exc:
        fallback = cfg.get("rerank.models.comparison_fallback")
        print(f"Comparison model {primary_choice!r} unavailable ({exc}); falling back to {fallback!r}")
        return fallback


def main():
    set_seed()
    cfg = load_config()
    corpus, queries, qrels, pool = load_processed_and_pool(cfg)
    corpus_lookup = corpus.set_index("doc_id")

    pool_sizes = cfg.get("retrieval.candidate_pool_sizes", [5, 10, 20, 50])
    metrics_k = cfg.get("evaluation.cutoff", 5)
    warmup = cfg.get("timing.warmup_runs", 3)
    repeats = cfg.get("timing.timed_repeats", 5)

    model_primary = cfg.get("rerank.models.primary")
    model_comparison = resolve_comparison_model(cfg)
    models = {"ms_marco_L6": model_primary, "medcpt": model_comparison}

    grid_rows = []

    for model_label, model_name in models.items():
        print(f"\n=== Loading {model_label}: {model_name} ===")
        reranker = CrossEncoder(model_name=model_name)

        for k in pool_sizes:
            print(f"  Reranking at K={k} ...")
            timer = StageTimer()
            with timer.stage("rerank"):
                run = rerank_at_k(reranker, queries, pool, corpus_lookup, k)

            per_query = evaluate_run(run, qrels, k=min(metrics_k, k))
            agg = aggregate(per_query)

            sample_row = queries.iloc[0]
            sample_candidates = [
                (doc_id, corpus_lookup.loc[doc_id, "text_raw"])
                for doc_id in pool.get(sample_row.query_id, [])[:k]
                if doc_id in corpus_lookup.index
            ]
            _, timing = measure(
                lambda: reranker.rerank(sample_row.query_raw, sample_candidates, top_k=k),
                f"{model_label}_k{k}",
                warmup,
                repeats,
            )

            grid_rows.append(
                {
                    "model": model_label,
                    "model_name": model_name,
                    "k": k,
                    f"ndcg@{metrics_k}": agg.get(f"ndcg@{min(metrics_k, k)}", float("nan")),
                    "mrr": agg.get("mrr", float("nan")),
                    "n_queries": agg.get("n_queries", 0),
                    "latency_p50_ms": timing.p50_ms,
                    "latency_p95_ms": timing.p95_ms,
                }
            )
            print(f"    nDCG@{metrics_k}={grid_rows[-1][f'ndcg@{metrics_k}']:.4f}  "
                  f"p50={timing.p50_ms:.1f}ms  p95={timing.p95_ms:.1f}ms")

        del reranker

    grid_df = pd.DataFrame(grid_rows)

    tables_dir = cfg.path("output.tables_dir")
    grid_df.to_csv(tables_dir / "stage6_grid.csv", index=False)

    figures_dir = cfg.path("output.figures_dir")
    _plot_grid(grid_df, figures_dir, metrics_k)

    write_manifest(
        "task6_candidate_size_experiment",
        cfg.path("output.logs_dir"),
        config=cfg.as_dict(),
        extra={"pool_sizes": pool_sizes, "models": models, "n_queries": len(queries)},
    )

    print("\n=== Task 6: candidate-size x model grid ===")
    print(grid_df.to_string(index=False))
    print(f"\nSaved grid to {tables_dir}, figures to {figures_dir}")


def _plot_grid(grid_df: pd.DataFrame, figures_dir: Path, metrics_k: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ndcg_col = f"ndcg@{metrics_k}"

    fig, ax = plt.subplots(figsize=(6, 4))
    for model_label, group in grid_df.groupby("model"):
        group = group.sort_values("k")
        ax.plot(group["k"], group[ndcg_col], marker="o", label=model_label)
    ax.set_xlabel("Candidate pool size (K)")
    ax.set_ylabel(f"nDCG@{metrics_k}")
    ax.set_title(f"Figure: nDCG@{metrics_k} vs. candidate pool size K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "stage6_ndcg_vs_k.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for model_label, group in grid_df.groupby("model"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["latency_p50_ms"], marker="o", label=model_label)
    ax.set_xlabel("Candidate pool size (K)")
    ax.set_ylabel("Reranking latency, p50 (ms)")
    ax.set_title("Figure: reranking latency vs. candidate pool size K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "stage6_latency_vs_k.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
