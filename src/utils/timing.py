"""Latency measurement with warm-up and percentile reporting.

Why not just ``time.time()`` around one call: the first invocation of any transformer
includes model loading, lazy kernel compilation, and cache population. That single
outlier dominates a mean over a handful of runs, so a naive measurement reports the
cost of *starting* the system rather than the cost of *using* it.

We therefore discard N warm-up runs, then report the **median (p50) and p95** over the
timed repeats. Tasks 2, 5, and 6 all turn on the quality-versus-latency trade-off, so
these numbers carry marks and need to be defensible.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TimingResult:
    """Latency statistics for one measured operation, in milliseconds."""

    label: str
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    n_runs: int
    raw_ms: list[float] = field(repr=False, default_factory=list)

    def as_row(self) -> dict[str, Any]:
        """Flatten for a results table (raw samples excluded)."""
        return {
            "operation": self.label,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "n_runs": self.n_runs,
        }


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile.

    ``statistics.quantiles`` needs at least two points and cuts at fixed positions;
    with 5 repeats we want an exact p95, so we interpolate directly.
    """
    if not values:
        raise ValueError("cannot take a percentile of an empty sample")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def measure(
    fn: Callable[[], Any],
    label: str,
    warmup: int = 3,
    repeats: int = 5,
) -> tuple[Any, TimingResult]:
    """Time ``fn`` after warm-up. Returns ``(last_result, timing)``.

    The callable's return value is passed back so a measured call does not have to be
    run a second time to obtain its output.
    """
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)

    return result, TimingResult(
        label=label,
        p50_ms=_percentile(samples, 50),
        p95_ms=_percentile(samples, 95),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        n_runs=repeats,
        raw_ms=samples,
    )


class StageTimer:
    """Accumulates per-stage latencies across many queries.

    Tasks 5 and 6 need the *marginal* cost of one additional candidate, which an
    end-to-end total cannot reveal — retrieval cost is flat in K while reranking cost
    is linear in it. Recording the stages separately is what lets the report say which
    component the latency actually comes from.

    Usage::

        timer = StageTimer()
        for q in queries:
            with timer.stage("retrieval"):
                pool = retrieve(q)
            with timer.stage("rerank"):
                ranked = rerank(q, pool)
        timer.summary()
    """

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.records.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> list[dict[str, Any]]:
        """One row per stage, plus a total row summing the per-query stage costs."""
        rows = [
            TimingResult(
                label=name,
                p50_ms=_percentile(samples, 50),
                p95_ms=_percentile(samples, 95),
                mean_ms=statistics.fmean(samples),
                min_ms=min(samples),
                max_ms=max(samples),
                n_runs=len(samples),
            ).as_row()
            for name, samples in self.records.items()
        ]

        if len(self.records) > 1:
            n = min(len(s) for s in self.records.values())
            totals = [sum(s[i] for s in self.records.values()) for i in range(n)]
            rows.append(
                TimingResult(
                    label="total",
                    p50_ms=_percentile(totals, 50),
                    p95_ms=_percentile(totals, 95),
                    mean_ms=statistics.fmean(totals),
                    min_ms=min(totals),
                    max_ms=max(totals),
                    n_runs=n,
                ).as_row()
            )
        return rows

    def reset(self) -> None:
        self.records.clear()
