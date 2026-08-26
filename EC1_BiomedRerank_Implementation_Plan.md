# EC-1: Cross-Encoder Reranking for Biomedical Literature Retrieval
## End-to-end implementation plan, repo structure, and playbook

**Purpose:** get full marks, not "get it working." These are different projects.

**Companion documents:**
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design, data flow, component contracts, diagrams
- [`docs/RUBRIC_MAP.md`](docs/RUBRIC_MAP.md) — every brief requirement mapped to the artefact that satisfies it

---

## 0. Read this first — open questions on the brief

**0.1 The marks don't add up.** Task 1 (1) + Task 2 (2) + Task 3 (2) + Task 4 (2) +
Task 5 (2) + Task 6 (1) + Task 7 (1) = **11 marks**, but the header says Total: 10.
Worth one clarifying email — but **do not let it block day 1.** The effort allocation
barely changes whether Task 4 is worth 1 or 2. Send the email, start building.

**0.2 Course code mismatch.** Header says "Conversational AI (AIMLCZG521)"; the file
is named `AIMLZG521`. Confirm you are on the correct Assignment Set — the brief says
a wrong set is not evaluated at all. This one *is* worth confirming early.

**0.3 "Virtual Lab screenshots" appear in the penalty clause but nowhere in the
deliverables list.** Combined with *"Assignments developed or submitted using any
Python IDE other than the prescribed environment will not be considered for
grading"*, this is the single largest zero-risk in the assignment. Develop wherever
you like; **execute and capture in the prescribed environment.**

**Concrete de-risking, not just a warning:** confirm what "prescribed environment"
means *before* the final week. If it is a CPU-only sandbox without internet, the
`ncbi/MedCPT-Cross-Encoder` download will fail there. Therefore:

- Pre-cache **all** model weights and the dataset into `data/raw/` and a local
  `models/` directory, committed to the submission bundle or copied in manually.
- Keep `cross-encoder/ms-marco-MiniLM-L-12-v2` as the declared fallback comparison
  model — it is far smaller and may already be cached.
- Verify the pipeline runs with `HF_HUB_OFFLINE=1` **on your own machine first.**
  If it works offline locally, it will work in the sandbox.

---

## 1. Where the marks actually are

Roughly 1.5–2 of 11 marks are for code that runs. The rest is for *explanation,
justification, inference, limitations, and error analysis.* The brief says this five
separate times (General Instructions 6, 9, 10, 13, 16).

Practical consequence: the pipeline is ~200 lines of real logic. Budget **30% of
time on code, 70% on instrumentation + writing.** The failure mode is a beautiful
notebook with three sentences of analysis under each cell.

*Analogy:* this is graded like a lab report in chemistry, not like a coding
interview. Nobody gives marks for titrating correctly. They give marks for
explaining why the endpoint drifted.

---

## 2. Architectural decisions (make these once, lock them)

Full rationale in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Summary:

| Decision | Choice | Why |
|---|---|---|
| Dataset | **BEIR / SciFact** | Small enough for CPU, real qrels, biomedical, citable. |
| Backup dataset | NFCorpus (BEIR) or TREC-COVID subset | If SciFact download fails in the sandbox. Pre-cache in `data/raw/`. |
| Stage-1 retrieval | **BM25 (hand-implemented) + dense + hybrid via RRF** | Brief allows any one. Doing all three converts Task 2 from "did it" to "compared it" — that's where the analysis marks live. |
| Cross-encoder A | `cross-encoder/ms-marco-MiniLM-L-6-v2` | General-domain baseline. Fast on CPU. |
| Cross-encoder B | `ncbi/MedCPT-Cross-Encoder` (fallback `ms-marco-MiniLM-L-12-v2`) | Domain-matched vs domain-mismatched is a far better story than L-6 vs L-12. |
| Metrics | **Hand-implemented** P@5, R@5, MRR, nDCG@5 | Brief forbids opaque use of built-in pipelines. Unit-test against `pytrec_eval` and show the parity cell. |
| Notebook role | Thin orchestration layer over `src/` | Notebooks are unmergeable JSON; modules diff cleanly. |

### Confidence tags on dataset facts
- [Likely] SciFact in BEIR: ~5,183 abstracts, 300 test queries.
- [Certain] SciFact relevance labels are effectively **binary**, and most queries have **exactly one** relevant document.
- [Likely] BM25 is an unusually strong baseline on SciFact; expect a modest reranker gain, not a dramatic one.
- [Guessing] Exact numbers. **Do not quote any figure in the report that you did not produce yourself.**

### The binary-label insight — put this in the report, it earns marks

With one relevant document per query, the metrics do not mean what their names
suggest. State this *before* the numbers:

| Metric | Behaviour under 1 relevant doc |
|---|---|
| **P@5** | Mathematically capped at **0.20**. Report as fraction of achievable max. |
| **R@5** | Degenerates to a **hit-rate** (1.0 or 0.0). Moves in lockstep with whether MRR is defined. |
| **MRR** | The genuinely informative metric — *where* the answer landed. |
| **nDCG@5** | With binary gains, a **position-discount** measure only. |

Reporting "P@5 = 0.19" as if it were poor is a mistake graders notice. The Recall@5
half of this is the one most teams miss — do not present a hit-rate as if it were
recall.

---

## 3. The one idea that separates a 7 from a 10: the recall ceiling

A reranker can only reorder what stage 1 already retrieved.

*Analogy:* the bouncer can reshuffle the queue outside the club, but cannot admit
someone who never showed up.

So **before any reranking numbers, report Recall@K of the candidate pool** for
K = 5, 10, 20, 50, 100. That is your headroom. If Recall@20 = 0.85, no cross-encoder
on earth beats 0.85 with a pool of 20, and a reranked 0.80 is 94% of achievable —
not "80%, which sounds mediocre."

This single table reframes Tasks 5, 6, and 7 and gives a principled answer to "why
did reranking not help here?" — it separates *retrieval* failures (document absent
from the pool) from *reranking* failures (present but mis-scored). These have
different fixes, and conflating them is the most common analytical error in this
assignment.

---

## 4. Task-by-task implementation plan

### Task 1 — Data prep (1 mark)
**Build:** `src/data_prep.py`
- Load BEIR SciFact → corpus, queries, qrels.
- Dedup on normalized title+abstract hash; log how many removed.
- Handle empty/missing abstracts (title-only docs) — log count, state policy explicitly.
- Strip HTML/LaTeX artifacts, normalize unicode (NFKC), collapse whitespace.
- **Do not** lowercase/stem for the cross-encoder path. Keep two views: `text_raw`
  (transformer) and `text_norm` (BM25). Justify in writing — it shows you understand
  that BM25 and BERT want opposite preprocessing.
- Save to `data/processed/{corpus,queries,qrels}.parquet`.

**Write (this is the mark):** the four required discussion points, with *measured
evidence* rather than opinions:
- Token-length histogram of abstracts → sets up the 512-token truncation problem in Task 7.
- Count of abstracts containing ≥1 all-caps abbreviation (regex `\b[A-Z]{2,6}\b`) → quantifies the abbreviation challenge.
- 2–3 real ambiguous terms from *your own* corpus (an abbreviation with two distinct expansions across papers). Concrete examples beat generic statements about "medical terminology is hard."

### Task 2 — Initial retrieval (2 marks)
**Build:** `src/retrieval/{bm25,dense,hybrid}.py`
- Implement BM25 **from the formula** (k1=1.2, b=0.75), then assert your scores match
  `rank_bm25` to within tolerance. **Show that assertion cell** — it directly answers
  "must not simply use built-in APIs."
- Dense: `sentence-transformers/all-MiniLM-L6-v2` or a biomedical encoder; cosine over normalized embeddings.
- Hybrid: Reciprocal Rank Fusion, `1/(60 + rank)`.
- Latency via `src/utils/timing.py` — 3 warmup runs, 5 timed repeats, report **median
  and p95**, not mean (mean is distorted by first-call model loading).
- **Record latency per stage** (retrieval ms / rerank ms / total ms), not just
  aggregate. Tasks 5 and 6 both need the *marginal* cost per additional candidate,
  and only a per-stage split reveals it.

**Write:** advantages/limits per method; then show a *lexical mismatch* case (relevant
doc BM25 misses due to a synonym) and a *semantic drift* case (dense pulls a
topically-similar but irrelevant paper). One concrete example of each beats a page of prose.

### Task 3 — Cross-encoder reranking (2 marks)
**Build:** `src/rerank/cross_encoder.py`
- Use `AutoTokenizer` + `AutoModelForSequenceClassification` directly, **not**
  `CrossEncoder.predict()`. Print the tokenized `[CLS] query [SEP] doc [SEP]`, show
  `input_ids` for one example, extract the logit yourself. This is the "demonstrate
  conceptual understanding" requirement, cashed out.
- Log `n_tokens_before_truncation` and a `was_truncated` boolean per pair. Needed in Task 7.
- Batch size fixed (16), `torch.no_grad()`, `model.eval()`, seed pinned.

**Write:** all five required discussion points. For bi- vs cross-encoder use the
analogy: a bi-encoder is two people writing summaries in separate rooms then
comparing them; a cross-encoder is one person reading query and document side by
side. The bi-encoder precomputes document vectors offline — that's why it scales to
millions. The cross-encoder runs once *per pair*, so cost is O(K) per query and it
can never be stage 1.

### Task 4 — Before vs after (2 marks)

> **Requirement check:** the brief says *"at least five biomedical research
> queries"* — five **distinct queries**, not five cases that might come from two
> queries. Constrain the mining to one case per query ID, five distinct IDs, and say
> so explicitly in the notebook.

**Do not hand-hunt these.** Instrument it. `src/analysis/rank_deltas.py` computes for
every (query, doc): `rank_stage1`, `rank_ce`, `delta`, `is_relevant`, `was_truncated`,
then auto-buckets:

| Required case | Automated selection rule |
|---|---|
| Relevant promoted | `is_relevant` and `rank_stage1 > 5` and `rank_ce <= 3` |
| Irrelevant demoted | `not is_relevant` and `rank_stage1 <= 3` and `rank_ce > 5` |
| Minimal change | `max abs(delta)` over top-5 `<= 1` |
| Failure | per-query `nDCG@5_ce < nDCG@5_stage1` |
| Terminology/abbrev | query matches abbreviation regex, sorted by `abs(delta)` |

Then hand-annotate the top candidate from each bucket. **Machine finds them, human
explains them.** Include actual query text and truncated doc snippets in a table, and
for each: why the ranking changed, and whether the final ranking is more useful to
the user (the brief asks both).

### Task 5 — Evaluation (2 marks)
**Build:** `src/eval/metrics.py` (hand-rolled) + `significance.py`
- Run quantitative tables over **all 300 test queries** (it's cheap); use 10–15 for
  narrative walkthroughs. The brief's "at least 10" is a floor; clearing it by 20x is
  free credibility.
- Add **paired bootstrap** or Wilcoxon signed-rank on per-query nDCG@5 deltas.
- Report the per-query delta *distribution*, not just means. A mean gain of +0.03 from
  15% of queries improving a lot and 10% degrading is a completely different system
  from one improving everything slightly — saying so is exactly the "inference" the
  rubric wants.

**Write:** all four required points. Limitations: latency, no recall recovery,
domain shift from MS MARCO web queries to scientific claims, 512-token truncation,
and no score calibration across queries (logits are not comparable between queries).

### Task 6 — Candidate size experiment (1 mark)
Do **both** options — it's a 2×3 grid at one extra loop's cost:
- Pool sizes K ∈ {5, 10, 20, 50}
- Models ∈ {ms-marco-MiniLM-L-6, MedCPT (or L-12)}

> **Scope valve:** the brief says *"Alternatively"* — one option suffices. If time is
> short, cut to pool sizes only (1×4). Cut the model axis, never the analysis.

Two plots on a shared x-axis (K):
1. nDCG@5 vs K → expect **saturation**.
2. Latency vs K → expect **linear**.

The knee is your recommendation. *Analogy:* screening more résumés costs linearly,
but the best candidate is usually already in the first twenty; the 200th résumé costs
the same as the 2nd and almost never wins.

Hold everything else constant (hardware, batch size, seed, query set) and **state in
the report that you did** — the brief has an explicit Experimental Constraint clause
and graders look for it. Reproduce the fixed-factor table from
`docs/ARCHITECTURE.md` §7 verbatim.

### Task 7 — Error analysis (1 mark)

> **Requirement check:** the brief asks for *"at least five reranking failures"* —
> five **failure instances**, each analyzed. Not five distinct *causes*. Causes may
> repeat across instances; that repetition is itself a finding worth reporting.

Reuse the Task 4 instrumentation. Take the **5 worst per-query nDCG regressions** and
classify each against the brief's taxonomy. Three causes have hard instrumented
evidence:
- **Long-document truncation** → the `was_truncated` flag
- **Abbreviation ambiguity** → abbreviation regex hits + expansion-collision check
- **Negation/context misreading** → negation-cue detection near query terms

The remaining causes (topic mismatch, named-entity mismatch, insufficient context)
are argued from manual annotation. Evidence-backed beats hand-waved — and quantify
where you can ("N% of regressions involved a truncated document").

**Improvements section:** propose, and where cheap *prototype*, one fix — e.g.
sliding-window scoring over long abstracts with max-pooling instead of head
truncation. A single implemented fix with a before/after number is worth more than a
bulleted wishlist.

---

## 5. Requirements the brief states but a code-first plan silently drops

These are cheap and graders tick them off by heading. Missing them leaks marks
quietly.

**5.1 The 14 mandated notebook sections (General Instruction 7).** All fourteen must
be present as **explicit, labelled markdown headings** in the `.ipynb`:

1. Assignment title · 2. Student details · 3. Problem statement · 4. Dataset details
and source · 5. Tools and libraries used · 6. Code implementation · 7. Output
screenshots/results · 8. Explanation of the logic used · 9. Justification for the
chosen method/model/approach · 10. Inference drawn from the results · 11. Limitations
observed · 12. Possible improvements · 13. Final conclusion · 14. References

Substance spread through the notebook does not count if the grader cannot find the
heading.

**5.2 Architecture and workflow diagram (Instructions 8 and 15).** The brief demands
the architectural design and workflow be explained, and that *all* diagrams be
properly labelled and explained. It even hands you the pipeline in *Suggested
Workflow*. Render Figure 1 from `docs/ARCHITECTURE.md`, caption it, and **reference
it in the prose** — an uncited figure reads as decoration.

**5.3 A worked single-query trace.** One query walked end-to-end through every arrow
of the Suggested Workflow, showing intermediate output at each step: the query, the
BM25 top-K with scores, the constructed pair with `input_ids`, the logits, the
reordered list. This is the clearest possible evidence of conceptual understanding
and directly mirrors the diagram the brief drew.

**5.4 Scope/safety statement.** The brief says twice that this is literature
retrieval, not diagnosis or treatment advice. One labelled line near the top of the
notebook. Costs nothing; its absence is noticeable.

**5.5 Actionable Recommendations section.** The deliverables list asks for
"recommendations based on their experiments as an actionable report" and most teams
skip it. Make it decision-shaped: *"For a pool of 20 on CPU-only deployment, use
model X; below 10 queries/sec, reranking is not worth the latency."*

**5.6 Comparative tables and visualizations.** Named explicitly in the deliverables.
Every figure numbered, captioned, and cited in text.

---

## 6. Repository structure

The critical decision: **logic lives in `src/`, the notebook only orchestrates.**
`.ipynb` files are JSON with embedded outputs and execution counts — two people
editing one notebook produces unmergeable conflicts, every time. Python modules diff
cleanly.

```
biomed-rerank/
├── README.md                     # setup, how to run, notebook/report split
├── requirements.txt              # pinned versions
├── environment.md                # lab env notes + screenshot checklist
├── Makefile                      # make data | retrieve | rerank | eval | all
│
├── config/experiment.yaml        # seeds, model names, K values, batch size
│
├── data/
│   ├── raw/                      # gitignored; scripts/download_data.py
│   ├── processed/                # parquet outputs of Task 1
│   └── README.md                 # dataset provenance + citation
│
├── docs/
│   ├── ARCHITECTURE.md           # design, data flow, diagrams
│   └── RUBRIC_MAP.md             # brief requirement → artefact mapping
│
├── src/
│   ├── config.py                 # loads experiment.yaml, single source of truth
│   ├── data_prep.py              # Task 1
│   ├── retrieval/{bm25,dense,hybrid}.py
│   ├── rerank/cross_encoder.py   # Task 3, manual tokenization
│   ├── eval/{metrics,significance}.py
│   ├── experiments/{candidate_size,model_compare}.py
│   ├── analysis/{rank_deltas,error_taxonomy}.py
│   └── utils/{timing,seeding,manifest}.py
│
├── notebooks/EC1_BiomedRerank_FINAL.ipynb   # thin; imports from src/
├── tests/{test_metrics,test_bm25}.py        # parity vs pytrec_eval / rank_bm25
├── results/{tables/*.csv, figures/*.png, logs/run_manifest.json}
├── report/{report.md, figures/}
├── screenshots/virtual_lab/                 # timestamped, named by task
└── scripts/{download_data.py, build_submission.py}
```

**Git hygiene (if using version control):**
- One branch per task; PR + one reviewer.
- `nbstripout --install` so notebook outputs don't enter git during development.
  Re-run and commit outputs **once**, at the end, by the integration owner.
- Better: pair the notebook with `jupytext` (`.py:percent`) so notebook logic is
  reviewable in PRs.

---

## 7. Sequencing

**If working solo or in a pair**, ignore the ownership table below and follow the
day plan directly; cut Task 6's model axis first if time is short.

**If working as a team**, ownership must be **vertical** — each person owns code
*and* the writing for their task. Splitting "coders" from "writers" is how the
analysis ends up thin, and analysis is 70% of the marks.

| Owner | Tasks | Deliverable |
|---|---|---|
| A | Task 1 + `data_prep.py` | processed parquets + preprocessing write-up |
| B | Task 2 (BM25/dense/hybrid) | retrieval module + comparison analysis |
| C | Task 3 + Task 6 | reranker + candidate-size/model grid |
| D | Task 5 + `metrics.py` + tests | metric suite, significance, eval tables |
| E | Task 4 + Task 7 | case mining, error taxonomy, improvement prototype |
| Lead | integration, notebook, report, submission | final `.ipynb`, `report.pdf`, `.zip` |

**Hard dependency:** `metrics.py` must land first or Tasks 2, 3, and 7 are all
blocked. Build the ruler before the thing you're measuring, even though it feels
backwards.

**Suggested 10-day run:**
- D1–2: repo scaffold, config, download script, metrics + parity tests
- D3–4: Task 1 and Task 2 land
- D5: Task 3 lands; **end-to-end pipeline runs green**
- D6: Task 5 + Task 6 grids
- D7: Task 4 + Task 7 mining and annotation
- D8: report draft; every figure captioned and referenced in text
- D9: **full clean re-execution in the prescribed lab environment** + screenshots
- D10: PDF export, page-count trim, zip, submit

Do not compress days 9–10. That's where submissions die.

---

## 8. Page-count and submission traps

- A notebook printing 300 queries' worth of output renders to 80+ pages. **Print
  `df.head(10)`, write full tables to `results/tables/*.csv`.** Cap every print loop.
- The brief demands *"output displayed for every executed cell"* AND *"not more than
  30 pages."* These conflict unless planned for. **Resolution:** the notebook PDF is
  execution evidence (keep it lean); `report.pdf` is the ≤30-page analytical
  document. State the split in the README so the grader isn't confused.
- Every figure: numbered, captioned, and **referenced in the prose**.
- References: SciFact/BEIR papers, cross-encoder **model cards**, BM25 (Robertson &
  Zaragoza), sentence-transformers, Nogueira & Cho. Cite model cards, not just
  library names.

---

## 9. Pre-submission checklist

**Correctness / risk**
- [ ] Marks-total discrepancy raised with instructor
- [ ] Correct Assignment Set (PS2) confirmed
- [ ] Prescribed environment identified; pipeline verified with `HF_HUB_OFFLINE=1`
- [ ] All model weights + dataset pre-cached locally
- [ ] Notebook restarts and runs top-to-bottom with zero errors
- [ ] Final execution done in the prescribed environment; screenshots captured

**Evidence of understanding**
- [ ] BM25 validated against `rank_bm25`; parity cell visible
- [ ] Metrics validated against `pytrec_eval`; parity cell visible
- [ ] Manual tokenization shown (`input_ids` printed for one pair)
- [ ] Worked single-query end-to-end trace present

**Required content**
- [ ] All 14 mandated notebook sections present as labelled headings
- [ ] Architecture/workflow diagram rendered, captioned, cited in prose
- [ ] Scope statement (retrieval, not diagnosis) present
- [ ] Recall-ceiling table present and referenced in the analysis
- [ ] Binary-label metric caveat stated before the metric numbers
- [ ] Task 4: five **distinct queries**, one per required case, each explained
- [ ] Task 7: five **failure instances** classified with stated causes
- [ ] Experimental-constraint statement (fixed factors) written out
- [ ] Actionable Recommendations section present and decision-shaped

**Packaging**
- [ ] Report ≤ 30 pages, all figures captioned and cited
- [ ] References complete, including model cards
- [ ] Single `.zip` containing `.ipynb`, notebook PDF, report PDF, screenshots
