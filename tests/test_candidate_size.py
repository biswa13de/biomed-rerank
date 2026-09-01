"""Unit tests for Task 6's grid-experiment helpers."""

from __future__ import annotations

from unittest.mock import patch

from src.experiments.candidate_size import resolve_comparison_model


class _FakeConfig:
    def __init__(self, comparison: str, fallback: str):
        self._values = {
            "rerank.models.comparison": comparison,
            "rerank.models.comparison_fallback": fallback,
        }

    def get(self, path, default=None):
        return self._values.get(path, default)


def test_resolve_comparison_model_returns_primary_when_it_loads():
    cfg = _FakeConfig("ncbi/MedCPT-Cross-Encoder", "cross-encoder/ms-marco-MiniLM-L-12-v2")

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()):
        result = resolve_comparison_model(cfg)

    assert result == "ncbi/MedCPT-Cross-Encoder"


def test_resolve_comparison_model_falls_back_on_load_failure():
    cfg = _FakeConfig("ncbi/MedCPT-Cross-Encoder", "cross-encoder/ms-marco-MiniLM-L-12-v2")

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("no internet")):
        result = resolve_comparison_model(cfg)

    assert result == "cross-encoder/ms-marco-MiniLM-L-12-v2"
