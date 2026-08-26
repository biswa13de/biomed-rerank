# EC-1: Cross-Encoder Reranking for Biomedical Literature Retrieval

Conversational AI (AIMLCZG521) — EC-1 Assessment, Problem Statement 2.

A two-stage biomedical literature retrieval system: BM25 / dense / hybrid candidate
retrieval, followed by transformer cross-encoder reranking, evaluated on BEIR
SciFact.

## Documents

- [`EC1_BiomedRerank_Implementation_Plan.md`](EC1_BiomedRerank_Implementation_Plan.md) — task-by-task implementation plan
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design, data flow, diagrams, rationale
- [`docs/RUBRIC_MAP.md`](docs/RUBRIC_MAP.md) — every brief requirement mapped to its artefact; use this to check completeness before submitting
- [`S1_26_AIMLZG521_Assignment1_PS2_AT.md`](S1_26_AIMLZG521_Assignment1_PS2_AT.md) — the original assignment brief

## Scope

This system performs biomedical **literature retrieval** — ranking research
abstracts by relevance to a research query. It does **not** provide medical
diagnosis, treatment recommendations, or patient-specific medical advice.

## Environment

- Python **3.11** (pinned — PyTorch has no stable wheels for 3.14, which is what
  `python3` resolves to by default on this machine).
- Managed with [`uv`](https://github.com/astral-sh/uv).

```bash
make install     # creates .venv (Python 3.11) and installs requirements.txt
make test         # runs the parity tests (BM25 vs rank_bm25, metrics vs pytrec_eval)
```

Or manually:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/ -v
```

## Notebook vs report split

The brief requires *output for every executed cell* and, separately, a result
document of no more than 30 pages. With ~300 evaluation queries these conflict, so:

- `notebooks/EC1_BiomedRerank_FINAL.ipynb` — execution evidence. Every cell shows
  output, but print loops are capped (`df.head(10)`); full tables live in
  `results/tables/*.csv`.
- `report/report.pdf` — the ≤30-page analytical document: architecture, tables,
  figures, task-by-task analysis, error taxonomy, recommendations, references.

## Repository layout

```
config/            experiment.yaml — single source of truth for seeds/models/K/batch size
data/               raw/ (gitignored) and processed/ (parquet) datasets
src/                all pipeline logic; the notebook only orchestrates
  data_prep.py        Task 1
  retrieval/          bm25.py, dense.py, hybrid.py — Task 2
  rerank/             cross_encoder.py — Task 3
  eval/               metrics.py (hand-rolled), significance.py — Task 5
  experiments/        candidate_size.py, model_compare.py — Task 6
  analysis/           rank_deltas.py, error_taxonomy.py — Tasks 4 & 7
  utils/              timing.py, seeding.py, manifest.py
notebooks/          EC1_BiomedRerank_FINAL.ipynb
tests/              test_bm25.py, test_metrics.py — parity checks vs reference libs
results/            tables/, figures/, logs/run_manifest.json
report/             report.md / report.pdf
screenshots/virtual_lab/   timestamped execution evidence from the prescribed lab environment
scripts/            download_data.py, build_submission.py
```

## Design principles this repo follows

1. **Logic lives in `src/`, the notebook only orchestrates.** `.ipynb` files are JSON
   with embedded outputs — unmergeable between collaborators. Python modules diff
   cleanly.
2. **No opaque library calls where the brief asks for understanding.** BM25 is scored
   from the formula and asserted against `rank_bm25`; metrics are hand-rolled and
   asserted against `pytrec_eval`; the cross-encoder is invoked via
   `AutoTokenizer` + `AutoModelForSequenceClassification` directly, with the
   tokenized pair and logit shown, not `CrossEncoder.predict()`. The reference
   libraries are test oracles only, never part of the pipeline.
3. **The recall ceiling frames every downstream result.** See
   `docs/ARCHITECTURE.md` §6. Reranking can only reorder what stage 1 retrieved.
4. **Everything is measured, not asserted.** Latency uses warm-up + p50/p95, not a
   single wall-clock read. Truncation, abbreviation matches, and rank deltas are
   logged per (query, doc) pair so Tasks 4 and 7 mine cases from data rather than by
   hand.

## Status

Environment and evaluation foundation are built and verified:

- `src/config.py` — YAML-backed config, single source of truth
- `src/eval/metrics.py` — P@k, R@k, MRR, nDCG@k, recall ceiling, paired comparison — **verified exact parity with `pytrec_eval`** across randomized binary and graded-relevance test collections (`tests/test_metrics.py`, 14/14 passing)
- `src/retrieval/bm25.py` — Okapi BM25 from the formula — **verified exact parity with `rank_bm25`** on scores and top-k ranking (`tests/test_bm25.py`, 4/4 passing)
- `src/utils/{timing,seeding,manifest}.py` — reproducibility and measurement infrastructure

Remaining: `data_prep.py` (Task 1), `dense.py`/`hybrid.py` (Task 2), `cross_encoder.py`
(Task 3), `rank_deltas.py`/`error_taxonomy.py` (Tasks 4/7), `significance.py` and full
evaluation run (Task 5), `candidate_size.py`/`model_compare.py` (Task 6), the notebook,
and the report. See `EC1_BiomedRerank_Implementation_Plan.md` §4 for the task-by-task
plan and `docs/RUBRIC_MAP.md` for the full completeness checklist.
