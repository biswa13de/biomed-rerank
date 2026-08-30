"""Parity tests: our hand-implemented significance tests vs. scipy.stats.

Same pattern as tests/test_metrics.py — scipy is the oracle here and nowhere else.
"""

from __future__ import annotations

import random

import pytest

from src.eval.significance import paired_bootstrap_ci, wilcoxon_signed_rank

scipy_stats = pytest.importorskip("scipy.stats")


def test_wilcoxon_matches_scipy_on_random_deltas():
    rng = random.Random(7)
    deltas = [rng.gauss(0.05, 0.2) for _ in range(150)]

    ours = wilcoxon_signed_rank(deltas)
    ref = scipy_stats.wilcoxon([d for d in deltas if d != 0], correction=True)

    assert ours["p_value"] == pytest.approx(ref.pvalue, abs=1e-3)


def test_wilcoxon_all_zero_deltas_returns_p_one():
    result = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert result["p_value"] == 1.0
    assert result["n_nonzero"] == 0


def test_wilcoxon_detects_consistent_positive_shift():
    deltas = [0.1, 0.15, 0.2, 0.05, 0.12, 0.08, 0.11, 0.09, 0.13, 0.07]
    result = wilcoxon_signed_rank(deltas)
    assert result["p_value"] < 0.05


def test_bootstrap_ci_contains_true_mean_for_symmetric_data():
    rng = random.Random(3)
    deltas = [rng.gauss(0.03, 0.1) for _ in range(300)]
    result = paired_bootstrap_ci(deltas, n_samples=2000, seed=1)

    assert result["ci_lower"] < result["observed_mean_delta"] < result["ci_upper"]
    assert result["n_queries"] == 300


def test_bootstrap_ci_reproducible_with_fixed_seed():
    deltas = [0.1, -0.05, 0.2, 0.0, -0.1, 0.15]
    r1 = paired_bootstrap_ci(deltas, n_samples=500, seed=42)
    r2 = paired_bootstrap_ci(deltas, n_samples=500, seed=42)
    assert r1 == r2


def test_bootstrap_ci_all_positive_deltas_is_significant():
    deltas = [0.1, 0.2, 0.15, 0.3, 0.25]
    result = paired_bootstrap_ci(deltas, n_samples=2000, seed=1)
    assert result["ci_lower"] > 0
    assert result["significant"] is True
