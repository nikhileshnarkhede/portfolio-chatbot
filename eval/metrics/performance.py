"""
performance.py - D5 metrics: latency, throughput, cost, reliability.

All deterministic, all free. The spec weights this dimension 8/10 and puts it
last precisely because it is the easiest thing to measure and therefore the
easiest to over-weight - a config that is 200ms faster and slightly less
accurate is a bad trade, and a report that leads with latency invites it.

Two choices worth explaining.

**Percentiles use nearest-rank, not interpolation.** `statistics.quantiles`
interpolates between observations, which invents a p95 that no request actually
experienced and needs n >= 2 to work at all. Nearest-rank returns a real
measurement, is defined for n = 1, and is what you want when someone asks "how
slow was the slow one".

**Time-to-first-token is reported separately from total latency.** In a
streaming UI they are different products. A change that shortens the total call
while delaying the first token makes the app feel *slower*, and a report
carrying only a mean would call that an improvement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

NAN = float("nan")


def _clean(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # drop NaN
            out.append(f)
    return out


def percentile(values: Iterable[Any], p: float) -> float:
    """Nearest-rank percentile. `p` in 0-100. NaN when there is no data.

    Returns an observation that actually happened rather than an interpolated
    value between two that did.
    """
    data = sorted(_clean(values))
    if not data:
        return NAN
    if p <= 0:
        return data[0]
    rank = math.ceil((p / 100.0) * len(data))
    return data[min(max(rank, 1), len(data)) - 1]


def mean(values: Iterable[Any]) -> float:
    data = _clean(values)
    return sum(data) / len(data) if data else NAN


def stdev(values: Iterable[Any]) -> float:
    """Population standard deviation. Feeds the noise floor in §06."""
    data = _clean(values)
    if len(data) < 2:
        return NAN
    mu = sum(data) / len(data)
    return math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))


@dataclass
class LatencyStats:
    n: int = 0
    p50: float = NAN
    p95: float = NAN
    p99: float = NAN
    mean: float = NAN
    max: float = NAN

    def as_dict(self) -> dict:
        return asdict(self)


def latency_stats(values: Iterable[Any]) -> LatencyStats:
    data = _clean(values)
    return LatencyStats(
        n=len(data),
        p50=percentile(data, 50), p95=percentile(data, 95), p99=percentile(data, 99),
        mean=mean(data), max=max(data) if data else NAN,
    )


@dataclass
class PerformanceScores:
    """Everything D5 reports for one run."""
    ttft: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    node_latency: dict = field(default_factory=dict)

    input_tokens_total: int = 0
    output_tokens_total: int = 0
    tokens_per_turn_mean: float = NAN
    tokens_per_turn_p95: float = NAN
    cost_usd_total: float = 0.0
    cost_usd_per_turn: float = NAN
    usage_reported: float = NAN

    fallback_rate: float = NAN
    error_rate: float = NAN
    n_turns: int = 0

    cold_start: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def score(turns: list[dict], cold_start: dict | None = None) -> PerformanceScores:
    """Aggregate D5 over a run's turn records.

    `turns` are run-log records (from `observability.run_logger`), not live
    state - so a report can be recomputed from `runs/` without re-calling the
    model.
    """
    if not turns:
        return PerformanceScores(cold_start=cold_start or {})

    usages = [t.get("token_usage") or {} for t in turns]
    attempts = [t.get("model_attempts") or [] for t in turns]

    # Per-node percentiles, not just per-turn: a p95 on the whole turn cannot
    # tell you whether the slow tail is retrieval or generation.
    node_names: set[str] = set()
    for t in turns:
        node_names.update((t.get("timings") or {}).keys())
    node_latency = {
        node: latency_stats([(t.get("timings") or {}).get(node) for t in turns]).as_dict()
        for node in sorted(node_names)
    }

    token_totals = [float(u.get("total_tokens", 0) or 0) for u in usages]
    costs = [float(u.get("cost_usd", 0.0) or 0.0) for u in usages]

    return PerformanceScores(
        ttft=latency_stats([t.get("ttft_s") for t in turns]).as_dict(),
        latency=latency_stats([t.get("latency_s") for t in turns]).as_dict(),
        node_latency=node_latency,

        input_tokens_total=int(sum(float(u.get("input_tokens", 0) or 0) for u in usages)),
        output_tokens_total=int(sum(float(u.get("output_tokens", 0) or 0) for u in usages)),
        tokens_per_turn_mean=mean(token_totals),
        tokens_per_turn_p95=percentile(token_totals, 95),
        cost_usd_total=round(sum(costs), 6),
        cost_usd_per_turn=mean(costs),
        usage_reported=mean([1.0 if u.get("available") else 0.0 for u in usages]),

        fallback_rate=mean([1.0 if len(a) > 1 else 0.0 for a in attempts]),
        error_rate=mean([1.0 if t.get("error") else 0.0 for t in turns]),
        n_turns=len(turns),

        cold_start=cold_start or {},
    )


def render(scores: PerformanceScores) -> str:
    """The D5 block of a report."""
    def fmt(v: Any, unit: str = "", places: int = 2) -> str:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if f != f else f"{f:.{places}f}{unit}"

    lines = [
        f"  TTFT      p50 {fmt(scores.ttft.get('p50'), 's')}   "
        f"p95 {fmt(scores.ttft.get('p95'), 's')}   max {fmt(scores.ttft.get('max'), 's')}",
        f"  latency   p50 {fmt(scores.latency.get('p50'), 's')}   "
        f"p95 {fmt(scores.latency.get('p95'), 's')}   p99 {fmt(scores.latency.get('p99'), 's')}",
    ]
    for node, stats in scores.node_latency.items():
        lines.append(f"    {node:<14} p50 {fmt(stats.get('p50'), 's', 3)}   "
                     f"p95 {fmt(stats.get('p95'), 's', 3)}")

    lines += [
        f"  tokens    {scores.input_tokens_total} in / {scores.output_tokens_total} out   "
        f"mean {fmt(scores.tokens_per_turn_mean, '', 0)}/turn   "
        f"p95 {fmt(scores.tokens_per_turn_p95, '', 0)}",
        f"  cost      ${scores.cost_usd_total:.4f} total"
        + ("" if scores.cost_usd_total else "   (llm.pricing empty - free tier)"),
        f"  usage rep {fmt(scores.usage_reported, '', 3)}"
        + ("" if (scores.usage_reported or 0) >= 1.0
           else "   <- provider did not report usage on every turn"),
        f"  fallback  {fmt(scores.fallback_rate, '', 3)}   errors {fmt(scores.error_rate, '', 3)}",
    ]
    if scores.cold_start:
        parts = "   ".join(f"{k} {fmt(v, 's')}" for k, v in scores.cold_start.items())
        lines.append(f"  cold start {parts}")
    return "\n".join(lines)


__all__ = [
    "score", "render", "PerformanceScores", "LatencyStats",
    "percentile", "latency_stats", "mean", "stdev",
]
