"""Significance testing for paired per-query metric deltas — Task 5.

Both tests are hand-implemented (General Instruction 9): ``scipy.stats`` is used only
as a parity oracle in ``tests/test_significance.py``, never inside the pipeline.

Why two tests, not one
-----------------------
* **Paired bootstrap** resamples queries with replacement and recomputes the mean
  delta each time, giving a confidence interval on the *mean* improvement without
  assuming a distribution. Directly answers "is the mean gain reliably above zero."
* **Wilcoxon signed-rank** ranks the *magnitudes* of the deltas and tests whether
  positive and negative ranks are balanced. It is robust to the handful of huge
  outlier deltas a mean-based test is sensitive to, and only requires the
  distribution of deltas be symmetric under the null — a weaker assumption than the
  bootstrap's implicit i.i.d.-mean framing.

Reporting both, and the delta *distribution* alongside the summary statistic, is what
lets the report distinguish "everything improved a little" from "a few queries
improved a lot while others got worse" — the same mean can arise from either, and
they are different findings about the system (see ``compare_runs`` in ``metrics.py``).
"""

from __future__ import annotations

import random
from typing import Sequence


def paired_bootstrap_ci(
    deltas: Sequence[float],
    n_samples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval on the mean of ``deltas``.

    Resamples the (fixed) set of per-query deltas with replacement ``n_samples``
    times, computes the mean of each resample, and reports the percentile interval.
    The seed is fixed so the interval is reproducible run to run — a bootstrap CI
    that changes on every execution would undermine the "reproducible pipeline"
    requirement just as much as an unseeded model would.
    """
    if not deltas:
        raise ValueError("deltas must be non-empty")

    rng = random.Random(seed)
    n = len(deltas)
    observed_mean = sum(deltas) / n

    resample_means = []
    for _ in range(n_samples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(resample) / n)

    resample_means.sort()
    alpha = 1.0 - confidence_level
    lower_idx = int((alpha / 2) * n_samples)
    upper_idx = int((1 - alpha / 2) * n_samples) - 1
    upper_idx = min(upper_idx, n_samples - 1)

    ci_lower = resample_means[lower_idx]
    ci_upper = resample_means[upper_idx]

    return {
        "observed_mean_delta": observed_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "n_samples": n_samples,
        "n_queries": n,
        "significant": ci_lower > 0 or ci_upper < 0,
    }


def _rank_with_ties(values: Sequence[float]) -> list[float]:
    """Average (fractional) ranks, ascending, with ties sharing the mean rank —
    the standard convention Wilcoxon's rank-sum requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        for pos in range(i, j + 1):
            ranks[order[pos]] = avg_rank
        i = j + 1

    return ranks


def wilcoxon_signed_rank(deltas: Sequence[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on non-zero paired deltas.

    Computed from the definition: rank the absolute non-zero deltas, sum the ranks
    belonging to positive deltas (``W+``) and to negative deltas (``W-``), and
    normalize ``W+`` by the normal approximation to get a z-score and two-sided
    p-value. The normal approximation is standard practice once n is large enough to
    matter for significance (our n is 300, well past the usual n>=10 threshold for
    trusting it) and avoids needing exact signed-rank distribution tables.
    """
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"w_stat": 0.0, "z": 0.0, "p_value": 1.0, "n_nonzero": 0}

    abs_ranks = _rank_with_ties([abs(d) for d in nonzero])
    w_pos = sum(r for r, d in zip(abs_ranks, nonzero) if d > 0)
    w_neg = sum(r for r, d in zip(abs_ranks, nonzero) if d < 0)

    mean_w = n * (n + 1) / 4.0
    std_w = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5

    if std_w == 0:
        return {"w_stat": w_pos, "z": 0.0, "p_value": 1.0, "n_nonzero": n}

    # Continuity correction: subtract half a unit toward the mean before dividing.
    z = (w_pos - mean_w - 0.5 * (1 if w_pos > mean_w else -1)) / std_w
    p_value = _two_sided_normal_p(z)

    return {"w_stat": min(w_pos, w_neg), "z": z, "p_value": p_value, "n_nonzero": n}


def _two_sided_normal_p(z: float) -> float:
    """Two-sided p-value from a standard normal z-score, via the error function."""
    import math

    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
