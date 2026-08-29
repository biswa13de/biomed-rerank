"""Task 1 — Biomedical data cleaning and preparation.

Loads BEIR SciFact (corpus, queries, qrels), cleans it, and builds the two text views
that the rest of the pipeline depends on:

  * ``text_norm``  — lowercased, punctuation-stripped. What BM25 (src/retrieval/bm25.py)
                      scores against: it matches surface term forms, so anything that
                      makes two spellings of the same concept share a token form helps it.
  * ``text_raw``   — original casing and punctuation preserved. What the cross-encoder
                      (src/rerank/cross_encoder.py) tokenizes: a transformer's WordPiece
                      vocabulary and pretraining both assume natural text. Lowercasing
                      destroys the case signal that distinguishes a gene symbol like
                      *MDM2* from ordinary prose, and stemming produces token forms the
                      model never saw during pretraining.

BM25 and BERT want opposite preprocessing (see docs/ARCHITECTURE.md §4.1); keeping
both views costs one extra column and removes that confound entirely, rather than
compromising on a single "normalized enough" text field.

Run as a script to produce data/processed/{corpus,queries,qrels}.parquet:

    .venv/bin/python -m src.data_prep
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.config import load_config

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_INLINE_RE = re.compile(r"\$[^$]*\$")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\{[^}]*\}")
_ABBREV_RE = re.compile(r"\b[A-Z]{2,6}\b")


@dataclass
class CleaningStats:
    """Every number here is printed and discussed in the notebook (Task 1 write-up)."""

    n_corpus_raw: int = 0
    n_corpus_after_dedup: int = 0
    n_duplicates_removed: int = 0
    n_empty_or_missing_abstract: int = 0
    n_title_only_docs: int = 0
    n_html_artifacts_found: int = 0
    n_latex_artifacts_found: int = 0
    n_queries_raw: int = 0
    n_queries_final: int = 0
    n_qrels_raw: int = 0
    n_qrels_final: int = 0
    n_queries_with_multiple_relevant: int = 0
    doc_token_length_summary: dict[str, float] = field(default_factory=dict)
    n_docs_with_abbreviation: int = 0
    pct_docs_with_abbreviation: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _normalize_unicode(text: str) -> str:
    """NFKC-normalise so visually-identical characters (e.g. full-width digits,
    ligatures) collapse to one canonical form before anything else touches the text."""
    return unicodedata.normalize("NFKC", text)


def _strip_html_and_latex(text: str, stats: CleaningStats) -> str:
    """Remove HTML tags/entities and common inline LaTeX residue from abstracts.

    Scientific abstracts pulled from XML/HTML sources routinely carry stray markup
    (``<i>``, ``&amp;``) and LaTeX math (``$k_1$``) that a lexical or transformer
    model has no use for and that would otherwise leak into token counts.
    """
    if _HTML_TAG_RE.search(text) or "&" in text:
        stats.n_html_artifacts_found += 1
    if _LATEX_INLINE_RE.search(text) or _LATEX_CMD_RE.search(text):
        stats.n_latex_artifacts_found += 1

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _LATEX_INLINE_RE.sub(" ", text)
    text = _LATEX_CMD_RE.sub(" ", text)
    return text


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text_raw(text: str, stats: CleaningStats) -> str:
    """Build the ``text_raw`` view: cleaned, but casing and punctuation preserved.

    Order matters: unicode normalization first (so downstream regexes see one
    canonical form), then markup stripping (works on the normalized text), then
    whitespace collapse last (markup removal can itself introduce runs of spaces).
    """
    text = _normalize_unicode(text)
    text = _strip_html_and_latex(text, stats)
    return _collapse_whitespace(text)


def clean_text_norm(text_raw: str) -> str:
    """Build the ``text_norm`` view from an already-cleaned ``text_raw``: lowercased.

    Deliberately minimal — no stemming or stopword removal — so BM25 scoring in
    src/retrieval/bm25.py stays a direct application of the formula on a documented
    input, not on a second undocumented preprocessing decision.
    """
    return text_raw.lower()


def _title_abstract_hash(title: str, text_norm: str) -> str:
    """Dedup key: normalized title + abstract. Two records with different IDs but
    identical (title, abstract) after normalization are treated as duplicates —
    this catches near-duplicate ingestion artifacts that an exact-string dedup
    on the raw text would miss (e.g. differing only in HTML entities)."""
    key = f"{title.strip().lower()}||{text_norm.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_raw_scifact(cache_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load BEIR SciFact corpus, queries, and test qrels from HuggingFace.

    Returns raw (unprocessed) dataframes; cleaning happens in :func:`prepare_dataset`.
    """
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus", cache_dir=cache_dir)
    queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", cache_dir=cache_dir)
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test", cache_dir=cache_dir)

    corpus_df = corpus_ds.to_pandas().rename(
        columns={"_id": "doc_id", "title": "title", "text": "abstract"}
    )
    queries_df = queries_ds.to_pandas().rename(columns={"_id": "query_id", "text": "query_text"})
    qrels_df = qrels_ds.to_pandas().rename(
        columns={"query-id": "query_id", "corpus-id": "doc_id", "score": "relevance"}
    )

    # HF-side ids are typed inconsistently (str in corpus/queries, int in qrels) —
    # normalise to str everywhere so joins in later stages don't silently drop rows.
    corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    return corpus_df, queries_df, qrels_df


def prepare_corpus(corpus_df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Clean, dedup, and dual-view the corpus. See module docstring for the view split."""
    stats.n_corpus_raw = len(corpus_df)

    df = corpus_df.copy()
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")

    stats.n_empty_or_missing_abstract = int((df["abstract"].str.strip() == "").sum())
    stats.n_title_only_docs = stats.n_empty_or_missing_abstract  # explicit policy below

    # Policy (stated explicitly, per the brief's requirement to "decide policy
    # explicitly"): title-only records are KEPT, not dropped. A title alone is still
    # a valid, if weak, retrieval signal, and SciFact qrels may point at such records;
    # dropping them would silently reduce recall ceiling for reasons unrelated to the
    # retrieval or reranking methods under study.
    df["text_raw"] = [
        clean_text_raw(f"{t} {a}".strip(), stats) for t, a in zip(df["title"], df["abstract"])
    ]
    df["title"] = [clean_text_raw(t, stats) for t in df["title"]]
    df["text_norm"] = df["text_raw"].map(clean_text_norm)

    df["dedup_key"] = [
        _title_abstract_hash(t, tn) for t, tn in zip(df["title"], df["text_norm"])
    ]
    before = len(df)
    df = df.drop_duplicates(subset="dedup_key", keep="first").drop(columns="dedup_key")
    stats.n_duplicates_removed = before - len(df)
    stats.n_corpus_after_dedup = len(df)

    df["n_tokens"] = df["text_raw"].str.split().str.len()
    df["has_abbrev"] = df["text_raw"].map(lambda t: bool(_ABBREV_RE.search(t)))

    stats.n_docs_with_abbreviation = int(df["has_abbrev"].sum())
    stats.pct_docs_with_abbreviation = 100.0 * stats.n_docs_with_abbreviation / len(df)
    stats.doc_token_length_summary = {
        "mean": float(df["n_tokens"].mean()),
        "median": float(df["n_tokens"].median()),
        "p95": float(df["n_tokens"].quantile(0.95)),
        "max": float(df["n_tokens"].max()),
        "min": float(df["n_tokens"].min()),
        "pct_over_512_tokens": float(100.0 * (df["n_tokens"] > 512).mean()),
    }

    return df.reset_index(drop=True)[
        ["doc_id", "title", "text_raw", "text_norm", "n_tokens", "has_abbrev"]
    ]


def prepare_queries(
    queries_df: pd.DataFrame, qrels_df: pd.DataFrame, stats: CleaningStats
) -> pd.DataFrame:
    """Keep only queries that appear in the test qrels — the evaluation set (Task 5)."""
    stats.n_queries_raw = len(queries_df)

    test_query_ids = set(qrels_df["query_id"].unique())
    df = queries_df[queries_df["query_id"].isin(test_query_ids)].copy()

    df["query_text"] = df["query_text"].fillna("")
    df["query_raw"] = [clean_text_raw(q, stats) for q in df["query_text"]]
    df["query_norm"] = df["query_raw"].map(clean_text_norm)
    df["has_abbrev"] = df["query_raw"].map(lambda t: bool(_ABBREV_RE.search(t)))

    stats.n_queries_final = len(df)
    return df.reset_index(drop=True)[["query_id", "query_raw", "query_norm", "has_abbrev"]]


def prepare_qrels(
    qrels_df: pd.DataFrame, valid_doc_ids: set[str], stats: CleaningStats
) -> pd.DataFrame:
    """Drop qrel rows pointing at a doc_id that didn't survive corpus cleaning
    (e.g. removed as a duplicate) — otherwise the evaluator silently treats a
    dropped relevant document as "not retrieved" for a reason that has nothing to
    do with retrieval quality."""
    stats.n_qrels_raw = len(qrels_df)
    df = qrels_df[qrels_df["doc_id"].isin(valid_doc_ids)].reset_index(drop=True)
    stats.n_qrels_final = len(df)

    per_query = df.groupby("query_id").size()
    stats.n_queries_with_multiple_relevant = int((per_query > 1).sum())
    return df


_EXPANSION_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z-]{3,}")


def _nearby_expansion_word(text: str, span: tuple[int, int], window: int = 30) -> str | None:
    """Best-effort guess at the word an abbreviation stands for, used only to
    fingerprint *distinct* usages of the same abbreviation, not to actually decode it.

    Looks first at the word immediately after the abbreviation (common pattern:
    ``IL-6``, ``HIV-infected``) and falls back to the closest content word before it
    (common pattern: ``interleukin (IL)``). Short function words (``and``, ``the``,
    ``of``, ...) are filtered out via the length threshold in `_EXPANSION_WORD_RE`.
    """
    start, end = span
    following = text[end : end + window]
    match = _EXPANSION_WORD_RE.match(following.lstrip("-( ").strip())
    if match:
        return match.group().lower()

    preceding = text[max(0, start - window) : start]
    words = _EXPANSION_WORD_RE.findall(preceding)
    return words[-1].lower() if words else None


def find_ambiguous_abbreviations(
    corpus: pd.DataFrame,
    min_expansions: int = 2,
    min_occurrences: int = 3,
    max_occurrences: int = 40,
    top_n: int = 10,
):
    """Surface real abbreviations from *this* corpus that co-occur with more than one
    distinct nearby content word — concrete evidence for the "similar terminology in
    unrelated research" discussion point, grounded in our own data rather than a
    generic claim.

    ``min_occurrences``/``max_occurrences`` bound the search to abbreviations common
    enough to have multiple genuine usages but rare enough that the sample stays
    small enough to inspect by hand — a handful of occurrences with several distinct
    associated words is a much stronger ambiguity signal than a handful out of
    thousands (e.g. ``DNA``, which appears in almost every document and would swamp
    the ranking with contexts that are all describing the same thing).
    """
    from collections import defaultdict

    contexts: dict[str, set[str]] = defaultdict(set)
    occurrences: dict[str, int] = defaultdict(int)

    for text in corpus["text_raw"]:
        for match in _ABBREV_RE.finditer(text):
            abbrev = match.group()
            occurrences[abbrev] += 1
            expansion_word = _nearby_expansion_word(text, match.span())
            if expansion_word:
                contexts[abbrev].add(expansion_word)

    candidates = [
        {
            "abbreviation": abbrev,
            "n_occurrences": occurrences[abbrev],
            "n_distinct_nearby_words": len(ctx_set),
            "sample_nearby_words": sorted(ctx_set)[:8],
        }
        for abbrev, ctx_set in contexts.items()
        if len(ctx_set) >= min_expansions
        and min_occurrences <= occurrences[abbrev] <= max_occurrences
    ]
    # Rank by contexts-per-occurrence: an abbreviation where nearly every occurrence
    # brings a *new* nearby word is more likely to be genuinely polysemous than one
    # that repeats the same handful of words many times.
    candidates.sort(
        key=lambda c: c["n_distinct_nearby_words"] / c["n_occurrences"], reverse=True
    )
    return candidates[:top_n]


def prepare_dataset(cache_dir: str | None = None, processed_dir: str | None = None):
    """End-to-end Task 1 pipeline: load, clean, dual-view, dedup, save, report."""
    cfg = load_config()
    cache_dir = cache_dir or str(cfg.path("dataset.cache_dir"))
    processed_dir = processed_dir or str(cfg.path("dataset.processed_dir"))

    stats = CleaningStats()

    corpus_raw, queries_raw, qrels_raw = load_raw_scifact(cache_dir)
    corpus = prepare_corpus(corpus_raw, stats)
    queries = prepare_queries(queries_raw, qrels_raw, stats)
    qrels = prepare_qrels(qrels_raw, set(corpus["doc_id"]), stats)

    from pathlib import Path

    out_dir = Path(processed_dir)
    corpus.to_parquet(out_dir / "corpus.parquet", index=False)
    queries.to_parquet(out_dir / "queries.parquet", index=False)
    qrels.to_parquet(out_dir / "qrels.parquet", index=False)

    return corpus, queries, qrels, stats


if __name__ == "__main__":
    corpus, queries, qrels, stats = prepare_dataset()
    print("=== Task 1: Data Preparation Summary ===")
    for key, value in stats.as_dict().items():
        print(f"{key:38s} {value}")
    print(f"\nSaved: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")

    print("\n=== Sample ambiguous abbreviations (from this corpus) ===")
    for cand in find_ambiguous_abbreviations(corpus, top_n=5):
        print(cand)
