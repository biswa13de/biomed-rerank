# Rubric Map — every brief requirement → the artefact that satisfies it

Purpose: at submission time, walk this table top to bottom. Anything unticked is a
mark leaking. Requirements are quoted from `S1_26_AIMLZG521_Assignment1_PS2_AT.md`.

Status key: ☐ not started · ◐ in progress · ☑ done

---

## A. General Instructions (the ones that carry penalties)

| # | Requirement | Where it is satisfied | Status |
|---|---|---|---|
| 2 | Submit `.ipynb` **and** PDF of executed notebook | `scripts/build_submission.py` | ☐ |
| 3 | Output displayed for **every** executed cell | Notebook; capped prints (see §8 of plan) | ☐ |
| 4 | Executed in the **prescribed environment** | `environment.md` + `screenshots/virtual_lab/` | ☐ |
| 5 | Correct Assignment Set (PS2) | Confirm with instructor | ☐ |
| 6 | Detailed explanation, justification, inference **for every task** | Markdown cell after every task block | ☐ |
| 7 | 14 mandated sections (see §B below) | Notebook headings | ☐ |
| 8 | Architectural design, workflow, reasoning; chunking, retrieval, ranking/reranking, context construction, evaluation | `docs/ARCHITECTURE.md` → notebook §Architecture | ☑ drafted |
| 9 | Must **not** use built-in APIs/pipelines without explaining the process | Manual BM25 + manual tokenization + hand-rolled metrics, each with a visible parity cell | ☐ |
| 10 | No missing outputs / unclear explanations / missing Virtual Lab screenshots | Pre-submission checklist | ☐ |
| 11 | Final PDF report attached | `report/report.pdf` | ☐ |
| 12 | Verified full top-to-bottom execution | Restart-and-run-all before export | ☐ |
| 14 | Proper references for datasets, models, libraries, papers | `report.md` §References, incl. **model cards** | ☐ |
| 15 | All figures/tables/diagrams labelled **and explained** | Every figure numbered, captioned, cited in prose | ☐ |
| 16 | Conclusion summarizing observations, strengths, limitations, future work | Notebook §13 + `report.md` | ☐ |

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
| Collection of documents + natural-language queries | `data/processed/*.parquet` | ☐ |
| Dedup, missing fields, HTML removal, normalization, relevance labels | `src/data_prep.py` with logged counts | ☐ |
| Saved in structured format for indexing | Parquet | ☐ |
| **Report:** why preprocessing matters for biomedical retrieval | Markdown + measured evidence | ☐ |
| **Report:** challenges of long scientific abstracts | Token-length histogram | ☐ |
| **Report:** medical terminology, abbreviations, named entities | Abbreviation regex counts | ☐ |
| **Report:** similar terminology in unrelated research | 2–3 real ambiguous terms from **our** corpus | ☐ |

### Task 2 — Initial Search System (2 marks)

| Requirement | Artefact | Status |
|---|---|---|
| Retrieval pipeline (BM25 / TF-IDF / dense) | `src/retrieval/{bm25,dense,hybrid}.py` | ☐ |
| Retrieve Top-K and rank by score | Run parquet files | ☐ |
| **Record initial ranking and retrieval latency** | `results/tables/`, per-stage p50/p95 | ☐ |
| **Analysis:** advantages and limitations of chosen method | Markdown, three-method comparison | ☐ |
| **Analysis:** keyword mismatch and semantic mismatch | One concrete example of each | ☐ |
| **Analysis:** relevant-but-lower-ranked cases | Concrete example from run data | ☐ |

### Task 3 — Cross-Encoder Reranking (2 marks)

| Requirement | Artefact | Status |
|---|---|---|
| Pre-trained cross-encoder reranks candidates | `src/rerank/cross_encoder.py` | ☐ |
| Query-document pairs → relevance score → sort → Top-K | Pipeline + worked trace | ☐ |
| **Explain:** bi-encoder vs cross-encoder | Markdown + Figure 2 | ☐ |
| **Explain:** how cross-encoder models query-doc interaction | Printed `input_ids`, joint attention | ☐ |
| **Explain:** why cross-encoders are useful for reranking | Markdown | ☐ |
| **Explain:** why reranking is applied only to a small candidate set | Cost O(K) argument + latency data | ☐ |
| **Explain:** how biomedical terminology affects relevance scoring | Domain-shift discussion + MedCPT contrast | ☐ |

### Task 4 — Before vs After Reranking (2 marks)

**At least FIVE distinct queries** — one per required case.

| Required case | Query ID | Status |
|---|---|---|
| Relevant document promoted | | ☐ |
| Irrelevant document moved lower | | ☐ |
| Ranking changes very little | | ☐ |
| Failure — reranking does not improve relevance | | ☐ |
| Domain-specific terminology / abbreviation | | ☐ |
| For **each**: why the ranking changed | | ☐ |
| For **each**: whether the final ranking is more useful to the user | | ☐ |

### Task 5 — Retrieval and Reranking Evaluation (2 marks)

| Requirement | Artefact | Status |
|---|---|---|
| Precision@5 | `src/eval/metrics.py` (hand-rolled) | ☐ |
| Recall@5 | ″ | ☐ |
| MRR | ″ | ☐ |
| NDCG@5 | ″ | ☐ |
| **At least 10 queries**, initial vs reranked | All 300 (20× the floor) | ☐ |
| Parity check vs `pytrec_eval` | `tests/test_metrics.py` + visible cell | ☐ |
| Binary-label caveat stated **before** the numbers | Markdown | ☐ |
| Recall-ceiling table (K = 5,10,20,50,100) | `results/tables/recall_ceiling.csv` | ☐ |
| **Discuss:** limitations of cross-encoder reranking | Markdown | ☐ |
| **Discuss:** retrieval quality vs reranking latency | Per-stage latency table | ☐ |
| **Discuss:** where reranking helps significantly | Per-query delta distribution | ☐ |
| **Discuss:** where it provides little/no improvement | Tied to recall ceiling | ☐ |

### Task 6 — Candidate Size Experiment (1 mark)

| Requirement | Artefact | Status |
|---|---|---|
| Compare pool sizes (Top-5 / 10 / 20 …) | K ∈ {5,10,20,50} | ☐ |
| *(Optional alt.)* compare two cross-encoders | MiniLM-L6 vs MedCPT | ☐ |
| Effect on **ranking quality** | Figure: nDCG@5 vs K | ☐ |
| Effect on **inference latency** | Figure: latency vs K | ☐ |
| Explain trade-off: more candidates vs compute cost | Knee analysis + recommendation | ☐ |
| **Experimental Constraint** statement (fixed factors) | `ARCHITECTURE.md` §7 table, reproduced | ☐ |

### Task 7 — Reranking Error Analysis (1 mark)

**At least FIVE failure instances.** Causes may repeat.

| # | Failure (query ID) | Assigned cause | Evidence | Status |
|---|---|---|---|---|
| 1 | | | | ☐ |
| 2 | | | | ☐ |
| 3 | | | | ☐ |
| 4 | | | | ☐ |
| 5 | | | | ☐ |
| | Improvements suggested for the identified failures | | | ☐ |
| | *(Bonus)* one improvement prototyped with before/after number | | | ☐ |

---

## D. Deliverables

| Requirement | Artefact | Status |
|---|---|---|
| Python notebook with results | `notebooks/EC1_BiomedRerank_FINAL.ipynb` | ☐ |
| PDF of notebook with results | `submission/notebook.pdf` | ☐ |
| Documentation | `report/report.pdf` | ☐ |
| Comparative tables and visualizations | `results/{tables,figures}` | ☐ |
| Output examples with analysis | Task 4 + Task 7 case tables | ☐ |
| **Actionable recommendations report** | `report.md` §Recommendations | ☐ |
| Single `.zip` | `scripts/build_submission.py` | ☐ |
| Result document ≤ 30 pages | Page count verified after export | ☐ |

---

## E. Items the brief mentions only in passing (easy to miss)

| Item | Where | Status |
|---|---|---|
| Scope statement: retrieval, **not** diagnosis/treatment advice | Notebook top + `ARCHITECTURE.md` | ☑ in arch |
| Dataset documentation: source, #queries, #docs, label format, splits | Notebook §4 + `data/README.md` | ☐ |
| Virtual Lab screenshots | `screenshots/virtual_lab/` | ☐ |
| Architecture / workflow diagram, labelled and explained | Figure 1 | ☑ drafted |
| Worked single-query end-to-end trace | Notebook | ☐ |
