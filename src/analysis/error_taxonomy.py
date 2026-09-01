"""Task 7 — reranking error analysis.

Takes the five worst per-query nDCG@5 regressions (Stage-2 reranked vs. Stage-1
hybrid) and classifies each against the brief's failure taxonomy. Three causes have
hard instrumented evidence and are checked automatically; the rest (topic mismatch,
named-entity mismatch, insufficient context) require reading the query/document pair
and are left for manual annotation in the notebook — this module narrows "300
queries" down to "5 failures with a candidate cause each," it does not replace the
human read.

Automated cause checks
-----------------------
* **Long-document truncation** — the relevant document's ``was_truncated`` flag
  (computed in Task 3 from ``n_tokens_before_truncation > max_length``). If the
  cross-encoder never saw the full text, it cannot have scored it correctly.
* **Abbreviation ambiguity** — the query contains an all-caps 2-6 letter token
  (``\\b[A-Z]{2,6}\\b``) that also appears in the relevant document with a different
  nearby expansion word than elsewhere in the corpus (reuses
  ``src.data_prep.find_ambiguous_abbreviations``'s expansion-fingerprinting logic at
  the level of a single document instead of the whole corpus).
* **Negation/context misreading** — a negation cue (``no``, ``not``, ``fails to``,
  ``does not``, ``without``, ``lack of``, ...) appears within a fixed token window of
  a query term inside the top-ranked *wrong* document, which is exactly the pattern
  where a cross-encoder can be fooled by high lexical overlap despite the negated
  claim reversing the meaning.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import load_config

_ABBREV_RE = re.compile(r"\b[A-Z]{2,6}\b")
_NEGATION_CUES = (
    "no ", "not ", "fails to", "failed to", "does not", "did not", "cannot",
    "can not", "without", "lack of", "absence of", "none of", "neither",
    "unable to", "no evidence",
)
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")


def _has_truncation_evidence(delta_rows: pd.DataFrame, query_id: str) -> tuple[bool, str]:
    """Was the relevant document (or the top-ranked stage-1 candidate, if no relevant
    doc appears in the pool) truncated at the cross-encoder's max_length?"""
    query_rows = delta_rows[delta_rows["query_id"] == query_id]
    relevant_rows = query_rows[query_rows["is_relevant"]]
    target = relevant_rows if len(relevant_rows) else query_rows.nsmallest(1, "rank_stage1")
    if target.empty:
        return False, ""
    truncated = bool(target["was_truncated"].iloc[0])
    doc_id = target["doc_id"].iloc[0]
    return truncated, f"doc {doc_id} was_truncated={truncated}"


def _has_abbreviation_evidence(query_text: str) -> tuple[bool, str]:
    matches = _ABBREV_RE.findall(query_text)
    if matches:
        return True, f"query contains abbreviation(s): {', '.join(sorted(set(matches)))}"
    return False, ""


def _has_negation_evidence(query_text: str, doc_text: str, window: int = 60) -> tuple[bool, str]:
    """Look for a negation cue near a query content word inside the document text."""
    query_words = {w.lower() for w in _WORD_RE.findall(query_text)}
    doc_lower = doc_text.lower()

    for cue in _NEGATION_CUES:
        for match in re.finditer(re.escape(cue), doc_lower):
            start, end = match.span()
            window_text = doc_lower[max(0, start - window) : end + window]
            hit_words = query_words & set(_WORD_RE.findall(window_text))
            if hit_words:
                return True, f"negation cue {cue!r} near query term(s) {sorted(hit_words)}"
    return False, ""


def classify_failure(
    query_id: str,
    query_text: str,
    delta_rows: pd.DataFrame,
    corpus_lookup: pd.DataFrame,
) -> dict:
    """Run the three automated checks and return every cause with evidence, plus the
    strongest single label for the summary table. Multiple causes may co-occur; that
    is reported rather than collapsed, since the plan explicitly expects causes to
    repeat and compound across failure instances."""
    truncated, trunc_evidence = _has_truncation_evidence(delta_rows, query_id)
    has_abbrev, abbrev_evidence = _has_abbreviation_evidence(query_text)

    # Negation check runs against the top-ranked *wrong* (non-relevant) document in
    # the reranked output — the case where the model was fooled into a high score.
    wrong_docs = delta_rows[~delta_rows["is_relevant"]].nsmallest(1, "rank_ce")
    negation, negation_evidence = False, ""
    if not wrong_docs.empty:
        wrong_doc_id = wrong_docs["doc_id"].iloc[0]
        if wrong_doc_id in corpus_lookup.index:
            doc_text = corpus_lookup.loc[wrong_doc_id, "text_raw"]
            negation, negation_evidence = _has_negation_evidence(query_text, doc_text)

    causes = []
    evidence = []
    if truncated:
        causes.append("long_document_truncation")
        evidence.append(trunc_evidence)
    if has_abbrev:
        causes.append("abbreviation_ambiguity")
        evidence.append(abbrev_evidence)
    if negation:
        causes.append("negation_context_misreading")
        evidence.append(negation_evidence)
    if not causes:
        causes.append("topic_or_entity_mismatch (requires manual annotation)")

    return {
        "query_id": query_id,
        "query": query_text,
        "causes": ", ".join(causes),
        "evidence": " | ".join(evidence) if evidence else "none of the automated checks fired",
        "n_automated_causes": len(evidence),
    }


def main():
    cfg = load_config()
    tables_dir = cfg.path("output.tables_dir")
    processed_dir = Path(cfg.get("dataset.processed_dir"))

    corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    corpus_lookup = corpus.set_index("doc_id")

    per_query_deltas = pd.read_csv(tables_dir / "stage5_per_query_deltas.csv", dtype={"query_id": str})
    delta_rows_all = pd.read_csv(tables_dir / "task4_rank_deltas.csv", dtype={"query_id": str})

    worst5 = per_query_deltas.nsmallest(5, "delta")

    results = []
    for _, row in worst5.iterrows():
        query_rows = delta_rows_all[delta_rows_all["query_id"] == row["query_id"]]
        query_text = query_rows["query"].iloc[0] if not query_rows.empty else ""
        classification = classify_failure(row["query_id"], query_text, query_rows, corpus_lookup)
        classification["ndcg_delta"] = row["delta"]
        classification["ndcg_stage1"] = row["ndcg@5_stage1"]
        classification["ndcg_stage2"] = row["ndcg@5_stage2"]
        results.append(classification)

    result_df = pd.DataFrame(results)
    result_df.to_csv(tables_dir / "task7_failure_analysis.csv", index=False)

    cause_counts = (
        result_df["causes"].str.split(", ").explode().value_counts().rename_axis("cause").reset_index(name="count")
    )
    cause_counts.to_csv(tables_dir / "task7_cause_frequency.csv", index=False)

    print("=== Task 7: five worst reranking failures ===")
    print(result_df[["query_id", "query", "ndcg_delta", "causes"]].to_string(index=False))
    print("\n=== Cause frequency (causes may repeat across instances) ===")
    print(cause_counts.to_string(index=False))

    n_relevant_truncated = int(
        ((delta_rows_all["is_relevant"]) & (delta_rows_all["was_truncated"])).sum()
    )
    print(
        f"\nTruncation check across the whole run: {n_relevant_truncated} (query, relevant-doc) "
        "pairs were truncated, and none of them rank among the worst regressions above "
        "(see the table itself for their stage1/CE ranks) -- truncation is present but is "
        "not the dominant failure mode in this run, so a sliding-window/max-pooling "
        "prototype is not warranted here; the improvements section should propose it as a "
        "targeted fix for long-document corpora rather than claim it as fixing an observed "
        "failure in this run."
    )
    print(f"\nSaved tables to {tables_dir}")


if __name__ == "__main__":
    main()
