# Rubric Map — every brief requirement → the artefact that satisfies it

Purpose: at submission time, walk this table top to bottom. Anything unticked is a
mark leaking. Requirements are quoted from `S1_26_AIMLZG521_Assignment1_PS2_AT.md`.

Status key: ☐ not started · ◐ in progress · ☑ done

---

## A. General Instructions (the ones that carry penalties)

| # | Requirement | Where it is satisfied | Status |
|---|---|---|---|
| 2 | Submit `.ipynb` **and** PDF of executed notebook | `notebooks/EC1_BiomedRerank_FINAL.ipynb` (self-contained; export to PDF is the report — no separate report file) | ☑ |
| 3 | Output displayed for **every** executed cell | Notebook, executed top-to-bottom via `nbconvert --execute` | ☑ |
| 4 | Executed in the **prescribed environment** | Notebook is self-contained (no `src` imports); runs from dataset/model cache only | ☑ |
| 5 | Correct Assignment Set (PS2) | Confirm with instructor | ☐ |
| 6 | Detailed explanation, justification, inference **for every task** | Markdown discussion cell after every task section in the notebook | ☑ |
| 7 | 14 mandated sections (see §B below) | Notebook headings | ☑ |
| 8 | Architectural design, workflow, reasoning; chunking, retrieval, ranking/reranking, context construction, evaluation | `docs/ARCHITECTURE.md` → notebook §Architecture and Workflow | ☑ |
| 9 | Must **not** use built-in APIs/pipelines without explaining the process | Manual BM25 + manual tokenization + hand-rolled metrics, each inlined in the notebook with a visible parity cell (Tasks 2 and 5) | ☑ |
| 10 | No missing outputs / unclear explanations / missing Virtual Lab screenshots | Pre-submission checklist | ☐ |
| 11 | Final PDF report attached | PDF export of `notebooks/EC1_BiomedRerank_FINAL.ipynb` (the notebook IS the report; no separate `report.pdf`) | ☐ export pending |
| 12 | Verified full top-to-bottom execution | `jupyter nbconvert --to notebook --execute --inplace` | ☑ |
| 14 | Proper references for datasets, models, libraries, papers | Notebook §References, incl. **model cards** | ☑ |
| 15 | All figures/tables/diagrams labelled **and explained** | Every figure numbered, captioned, cited in the notebook's discussion prose | ☑ |
| 16 | Conclusion summarizing observations, strengths, limitations, future work | Notebook §Inference / §Limitations Observed / §Possible Improvements / §Final Conclusion | ☑ |

## B. The 14 mandated notebook sections (Instruction 7)

Each must exist as an **explicit labelled markdown heading**. Substance scattered
elsewhere does not count if the grader cannot find the heading.

| # | Section | Status |
|---|---|---|
| 1 | Assignment title | ☐ |
| 2 | Student details | ☐ |
| 3 | Problem statement | ☐ |
| 4 | Dataset details and source | ☐ |
| 5 | Tools and libraries used | ☐ |
| 6 | Code implementation | ☐ |
| 7 | Output screenshots / results | ☐ |
| 8 | Explanation of the logic used | ☐ |
| 9 | Justification for chosen method/model/approach | ☐ |
| 10 | Inference drawn from the results | ☐ |
| 11 | Limitations observed | ☐ |
| 12 | Possible improvements | ☐ |
| 13 | Final conclusion | ☐ |
| 14 | References | ☐ |

---

## C. Task requirements

### Task 1 — Data Cleaning and Preparation (1 mark)

| Requirement | Artefact | Status |
|---|---|---|
| Collection of documents + natural-language queries | Notebook §Task 1, loaded live from the HF SciFact cache | ☑ |
| Dedup, missing fields, HTML removal, normalization, relevance labels | Notebook §Task 1 (inlined `data_prep` logic, logged counts) | ☑ |
| Saved in structured format for indexing | In-notebook DataFrames (`corpus`, `queries`, `qrels`) | ☑ |
| **Report:** why preprocessing matters for biomedical retrieval | Notebook §Task 1 Discussion, measured evidence | ☑ |
| **Report:** challenges of long scientific abstracts | Notebook §Task 1, token-length summary | ☑ |
| **Report:** medical terminology, abbreviations, named entities | Notebook §Task 1, abbreviation regex counts | ☑ |
| **Report:** similar terminology in unrelated research | Notebook §Task 1, `find_ambiguous_abbreviations()` output — 2–3 real ambiguous terms from **our** corpus | ☑ |

### Task 2 — Initial Search System (2 marks)

| Requirement | Artefact | Status |
|---|---|---|
| Retrieval pipeline (BM25 / TF-IDF / dense) | Notebook §Task 2 (BM25, DenseRetriever, HybridRetriever inlined and run live) | ☑ |
| Retrieve Top-K and rank by score | Notebook §Task 2, `bm25_run`/`dense_run`/`hybrid_run` computed over all 300 queries | ☑ |
| **Record initial ranking and retrieval latency** | Notebook §Task 2, `stage1_latency` table (p50/p95) | ☑ |
| **Analysis:** advantages and limitations of chosen method | Notebook §Task 2 Discussion, three-method comparison | ☑ |
| **Analysis:** keyword mismatch and semantic mismatch | Notebook §Task 2, `lexical_mismatch`/`semantic_drift` tables | ☑ |
| **Analysis:** relevant-but-lower-ranked cases | Notebook §Task 2 Discussion, from `stage1_metrics`/`stage1_ceiling` | ☑ |

### Task 3 — Cross-Encoder Reranking (2 marks)

| Requirement | Artefact | Status |
|---|---|---|
| Pre-trained cross-encoder reranks candidates | Notebook §Task 3 (`CrossEncoder` class inlined, manual `AutoTokenizer`/`AutoModelForSequenceClassification`) | ☑ |
| Query-document pairs → relevance score → sort → Top-K | Notebook §Task 3, `rerank_run` computed live + worked single-query trace | ☑ |
| **Explain:** bi-encoder vs cross-encoder | Notebook §Task 3 intro + Discussion, Figure 2 | ☑ |
| **Explain:** how cross-encoder models query-doc interaction | Notebook §Task 3, printed `input_ids`/`tokens` from the live trace | ☑ |
| **Explain:** why cross-encoders are useful for reranking | Notebook §Task 3 Discussion | ☑ |
| **Explain:** why reranking is applied only to a small candidate set | Notebook §Task 3 Discussion, O(K) argument + `stage2_latency` table | ☑ |
| **Explain:** how biomedical terminology affects relevance scoring | Notebook §Task 3 Discussion (deferred) → answered quantitatively in §Task 6 | ☑ |

### Task 4 — Before vs After Reranking (2 marks)

**At least FIVE distinct queries** — one per required case.

Auto-selected live in notebook §Task 4 (inlined `build_rank_delta_table` /
`select_task4_cases` logic) — see the `selected_cases` table and per-case evidence
cells; exact query ids may shift run-to-run since results are computed fresh, not
loaded from a saved CSV.

| Required case | Status |
|---|---|
| Relevant document promoted | ☑ auto-selected live |
| Irrelevant document moved lower | ☑ auto-selected live |
| Ranking changes very little | ☑ auto-selected live |
| Failure — reranking does not improve relevance | ☑ auto-selected live |
| Domain-specific terminology / abbreviation | ☑ auto-selected live |
| For **each**: why the ranking changed | ☐ needs hand annotation in notebook §Task 4 Discussion |
| For **each**: whether the final ranking is more useful to the user | ☐ needs hand annotation in notebook §Task 4 Discussion |

### Task 5 — Retrieval and Reranking Evaluation (2 marks)

| Requirement | Artefact | Status |
|---|---|---|
| Precision@5 | Notebook §Task 2 (hand-rolled, reused in §Task 5) | ☑ |
| Recall@5 | ″ | ☑ |
| MRR | ″ | ☑ |
| NDCG@5 | ″ | ☑ |
| **At least 10 queries**, initial vs reranked | All 300 (20× the floor) — notebook §Task 5, `stage5_metrics` table | ☑ |
| Parity check vs `pytrec_eval` | `tests/test_metrics.py` + notebook §Task 5 visible parity cell | ☑ |
| Paired significance test (bootstrap + Wilcoxon) | Notebook §Task 5 (inlined bootstrap CI + Wilcoxon), parity vs scipy in `tests/test_significance.py` and a visible notebook parity cell | ☑ |
| Binary-label caveat stated **before** the numbers | Notebook §Dataset Details and §Task 5, printed before every metric table | ☑ |
| Recall-ceiling table (K = 5,10,20,50,100) | Notebook §Task 5, `stage5_ceiling` table | ☑ |
| **Discuss:** limitations of cross-encoder reranking | Notebook §Task 5 Discussion | ☑ |
| **Discuss:** retrieval quality vs reranking latency | Notebook §Task 5 Discussion, referencing `stage2_latency`/`stage1_latency` | ☑ |
| **Discuss:** where reranking helps significantly | Notebook §Task 5 Discussion, live-computed improved/degraded percentages and significance result (numbers computed fresh each run, not hardcoded) | ☑ |
| **Discuss:** where it provides little/no improvement | Notebook §Task 5 Discussion, tied to the live `stage5_ceiling` table | ☑ |

### Task 6 — Candidate Size Experiment (1 mark)

Notebook §Task 6 (inlined grid logic, run live) — `stage6_grid` table, Figures 3 & 4
rendered inline as embedded images.

| Requirement | Artefact | Status |
|---|---|---|
| Compare pool sizes (Top-5 / 10 / 20 …) | K ∈ {5,10,20,50}, notebook §Task 6 | ☑ |
| *(Optional alt.)* compare two cross-encoders | MiniLM-L6 vs MedCPT — did both axes | ☑ |
| Effect on **ranking quality** | Notebook §Task 6, Figure 3 (nDCG@5 vs K) + live `_pivot_ndcg` table | ☑ |
| Effect on **inference latency** | Notebook §Task 6, Figure 4 (latency vs K) + live `_pivot_latency` table | ☑ |
| Explain trade-off: more candidates vs compute cost | Notebook §Task 6 Discussion, knee analysis + recommendation | ☑ |
| **Experimental Constraint** statement (fixed factors) | Notebook §Task 6 intro, table reproduced from `ARCHITECTURE.md` §7 | ☑ |

### Task 7 — Reranking Error Analysis (1 mark)

**At least FIVE failure instances.** Causes may repeat.

Notebook §Task 7 (inlined `classify_failure` / cause-check logic, run live) —
`task7_failures` and `task7_cause_freq` tables. Takes the 5 worst per-query nDCG@5
regressions from §Task 5's live-computed `per_query_deltas`; exact query ids may
shift run-to-run since results are computed fresh, not loaded from a saved CSV.

| # | Assigned cause | Status |
|---|---|---|
| 1–5 (five worst live regressions) | Automated causes (`long_document_truncation`, `abbreviation_ambiguity`, `negation_context_misreading`) checked live; unmatched cases flagged `topic_or_entity_mismatch (requires manual annotation)` | ☑ auto-classified live |
| | Manual annotation of any `topic_or_entity_mismatch` cases | ☐ needs hand annotation in notebook §Task 7 Discussion |
| | Improvements suggested for the identified failures | ☑ notebook §Task 7 Discussion |
| | *(Bonus)* one improvement prototyped with before/after number | Skipped, documented instead: the truncation-check cell in §Task 7 reports how many (query, relevant-doc) pairs were truncated across the whole run and whether any rank among the worst regressions — if none do, a sliding-window prototype is reported as a ruled-out cause for this run, not claimed as a fix | ☑ documented live negative-finding check |

---

## D. Deliverables

| Requirement | Artefact | Status |
|---|---|---|
| Python notebook with results | `notebooks/EC1_BiomedRerank_FINAL.ipynb` — self-contained, executed top-to-bottom | ☑ |
| PDF of notebook with results | PDF export of `notebooks/EC1_BiomedRerank_FINAL.ipynb` (this notebook IS the report — no separate report file) | ☐ export pending |
| Documentation | Notebook itself (14 mandated sections; no separate `report.pdf`) | ☑ |
| Comparative tables and visualizations | In-notebook tables and embedded figures (Tasks 2, 3, 5, 6) | ☑ |
| Output examples with analysis | Notebook §Task 4 + §Task 7 case tables | ☑ |
| **Actionable recommendations report** | Notebook §Actionable Recommendations | ☑ |
| Single `.zip` | `scripts/build_submission.py` | ☐ |
| Result document ≤ 30 pages | Page count verified after PDF export | ☐ |

---

## E. Items the brief mentions only in passing (easy to miss)

| Item | Where | Status |
|---|---|---|
| Scope statement: retrieval, **not** diagnosis/treatment advice | Notebook top + `ARCHITECTURE.md` | ☑ in arch |
| Dataset documentation: source, #queries, #docs, label format, splits | Notebook §4 + `data/README.md` | ☐ |
| Virtual Lab screenshots | `screenshots/virtual_lab/` | ☐ |
| Architecture / workflow diagram, labelled and explained | Figure 1 | ☑ drafted |
| Worked single-query end-to-end trace | Notebook | ☐ |
