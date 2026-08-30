"""Task 5 driver — full evaluation of Stage-1 hybrid vs. Stage-2 reranked over all 300
SciFact test queries, with paired significance testing on the per-query nDCG@5 deltas.

Read this before the numbers below (binary-label caveat, per docs/ARCHITECTURE.md and
src/eval/metrics.py's module docstring): SciFact qrels are binary and almost every
query has exactly one relevant document, so P@5 is capped at 0.20, R@5 degenerates to
a hit-rate, and nDCG@5 reduces to a position-discount measure. Read every number below
against that ceiling, not against a 0-to-1 scale.

Produces:
    results/tables/stage5_metrics_full.csv       hybrid vs reranked, all 300 queries
    results/tables/stage5_per_query_deltas.csv   per-query nDCG@5 for both, plus delta
    results/tables/stage5_significance.json      bootstrap CI + Wilcoxon on the deltas
    results/tables/stage5_recall_ceiling.csv     Recall@K carried forward from Stage 1
    results/logs/run_manifest.json               appended with this run's provenance

Run with:  .venv/bin/python -m src.eval.run_stage5
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.eval.metrics import aggregate, compare_runs, evaluate_run, ndcg_at_k
from src.eval.significance import paired_bootstrap_ci, wilcoxon_signed_rank
from src.utils.manifest import write_manifest


def load_runs_and_qrels(cfg):
    processed_dir = Path(cfg.get("dataset.processed_dir"))
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

    return qrels, load_run("hybrid"), load_run("rerank_ms_marco")


def main():
    cfg = load_config()
    qrels, stage1_run, stage2_run = load_runs_and_qrels(cfg)

    top_k_pool = cfg.get("retrieval.top_k_default", 20)
    stage1_run = {qid: docs[:top_k_pool] for qid, docs in stage1_run.items()}
    metrics_k = cfg.get("evaluation.cutoff", 5)

    per_query_stage1 = evaluate_run(stage1_run, qrels, k=metrics_k)
    per_query_stage2 = evaluate_run(stage2_run, qrels, k=metrics_k)

    agg_df = pd.DataFrame(
        {
            "stage1_hybrid": aggregate(per_query_stage1),
            "stage2_reranked": aggregate(per_query_stage2),
        }
    ).T.reset_index().rename(columns={"index": "method"})

    metric_name = f"ndcg@{metrics_k}"
    comparison = compare_runs(per_query_stage1, per_query_stage2, metric=metric_name)

    shared_qids = sorted(set(per_query_stage1) & set(per_query_stage2))
    per_query_rows = [
        {
            "query_id": qid,
            f"{metric_name}_stage1": per_query_stage1[qid][metric_name],
            f"{metric_name}_stage2": per_query_stage2[qid][metric_name],
            "delta": per_query_stage2[qid][metric_name] - per_query_stage1[qid][metric_name],
        }
        for qid in shared_qids
    ]
    per_query_df = pd.DataFrame(per_query_rows)
    deltas = per_query_df["delta"].tolist()

    bootstrap_result = paired_bootstrap_ci(
        deltas,
        n_samples=cfg.get("evaluation.bootstrap_samples", 10000),
        confidence_level=cfg.get("evaluation.confidence_level", 0.95),
    )
    wilcoxon_result = wilcoxon_signed_rank(deltas)

    significance = {
        "metric": metric_name,
        "comparison": comparison,
        "paired_bootstrap": bootstrap_result,
        "wilcoxon_signed_rank": wilcoxon_result,
    }

    ceiling_df = pd.read_csv(cfg.path("output.tables_dir") / "stage1_recall_ceiling.csv")
    ceiling_df = ceiling_df[ceiling_df["method"] == "hybrid"]

    tables_dir = cfg.path("output.tables_dir")
    agg_df.to_csv(tables_dir / "stage5_metrics_full.csv", index=False)
    per_query_df.to_csv(tables_dir / "stage5_per_query_deltas.csv", index=False)
    (tables_dir / "stage5_significance.json").write_text(json.dumps(significance, indent=2), encoding="utf-8")
    ceiling_df.to_csv(tables_dir / "stage5_recall_ceiling.csv", index=False)

    write_manifest(
        "task5_evaluation",
        cfg.path("output.logs_dir"),
        config=cfg.as_dict(),
        extra={"n_queries": len(shared_qids), "metric": metric_name, **comparison},
    )

    print("=== Task 5: Full evaluation (all 300 queries) ===")
    print("\nBinary-label caveat: SciFact qrels are binary with ~1 relevant doc/query.")
    print("P@5 is capped at 0.20; R@5 is a hit-rate; nDCG@5 is a position-discount measure.")
    print("Read the numbers below against those ceilings, not a 0-1 scale.\n")
    print(agg_df.to_string(index=False))

    print(f"\n=== Per-query {metric_name} delta distribution (stage2 - stage1) ===")
    print(f"n_queries={comparison['n_queries']}  improved={comparison['n_improved']} "
          f"({comparison['pct_improved']:.1f}%)  degraded={comparison['n_degraded']} "
          f"({comparison['pct_degraded']:.1f}%)  unchanged={comparison['n_unchanged']}")
    print(f"mean_delta={comparison['mean_delta']:.4f}")

    print("\n=== Significance (paired bootstrap, 95% CI on mean delta) ===")
    print(f"observed_mean={bootstrap_result['observed_mean_delta']:.4f}  "
          f"CI=[{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]  "
          f"significant={bootstrap_result['significant']}")

    print("\n=== Significance (Wilcoxon signed-rank) ===")
    print(f"z={wilcoxon_result['z']:.3f}  p={wilcoxon_result['p_value']:.4f}  "
          f"n_nonzero={wilcoxon_result['n_nonzero']}")

    print(f"\nRecall ceiling (hybrid pool, carried from Stage 1):")
    print(ceiling_df[["k", "recall@k"]].to_string(index=False))

    print(f"\nSaved tables to {tables_dir}")


if __name__ == "__main__":
    main()
