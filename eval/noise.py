"""
noise.py - measure how much the system moves when nothing changes.

Spec §06. This runs before any experiment, and it is the difference between an
evaluation and a dashboard.

Run the same config N times and compute the standard deviation of every metric.
That sigma is the noise floor; twice it is the minimum detectable effect. Any
reported improvement smaller than the MDE is a coin flip you have chosen to
believe.

Concretely, on a 22-case suite one case flipping moves a mean by 0.045 - so an
observed "+0.056 hit@k" may be exactly one question changing its mind. Until
sigma is measured, nobody can tell the difference, and the honest label for such
a delta is *inconclusive* rather than a green arrow.

The result feeds two places: `gate.check_regression` uses sigma as the
tolerance, and the dashboard uses the MDE to decide whether to render a delta as
a movement or as noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .metrics.performance import mean, stdev

#: Metrics worth tracking variance on. Cost and token counts are excluded: they
#: vary with answer length rather than with system quality, so their sigma
#: describes verbosity, not stability.
TRACKED = (
    "route_correct", "hit_at_k", "recall_at_k", "mrr", "ndcg_at_k",
    "type_precision", "chunk_coverage",
    "entity_grounding", "fabricated_entity_rate", "faithfulness", "relevancy",
    "fact_coverage", "context_overlap", "first_person",
    "attempted_forged_url", "refusal_correct", "consistency",
    "injection_success", "persona_break_rate",
    "error_rate", "ttft_p95", "latency_p50",
)


@dataclass
class NoiseFloor:
    n_repeats: int
    dataset: str
    means: dict[str, float] = field(default_factory=dict)
    sigmas: dict[str, float] = field(default_factory=dict)
    mde: dict[str, float] = field(default_factory=dict)
    per_run: list[dict] = field(default_factory=list)
    sigma_multiple: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_repeats": self.n_repeats, "dataset": self.dataset,
            "means": self.means, "sigmas": self.sigmas, "mde": self.mde,
            "sigma_multiple": self.sigma_multiple,
        }

    def is_significant(self, metric: str, delta: float) -> bool:
        """Does a delta exceed the minimum detectable effect for this metric?"""
        threshold = self.mde.get(metric)
        return threshold is not None and abs(delta) > threshold

    def render(self) -> str:
        lines = [
            f"NOISE FLOOR   {self.dataset}   {self.n_repeats} identical runs",
            "",
            f"{'metric':<26}{'mean':>10}{'sigma':>10}{'MDE':>10}   stability",
            "-" * 72,
        ]
        for metric in TRACKED:
            if metric not in self.means:
                continue
            mu, sd, mde = self.means[metric], self.sigmas.get(metric), self.mde.get(metric)
            if sd is None or sd != sd:
                lines.append(f"{metric:<26}{mu:>10.4g}{'n/a':>10}{'n/a':>10}")
                continue
            if sd == 0:
                note = "deterministic"
            elif sd < 0.02:
                note = "stable"
            elif sd < 0.05:
                note = "noisy"
            else:
                note = "VERY NOISY"
            lines.append(f"{metric:<26}{mu:>10.4g}{sd:>10.4g}{mde:>10.4g}   {note}")

        volatile = [m for m in TRACKED
                    if (self.sigmas.get(m) or 0) >= 0.05]
        if volatile:
            lines += ["", "These move by more than 0.05 between identical runs. Treat any",
                      "reported improvement in them below that size as noise:",
                      "  " + ", ".join(volatile)]
        deterministic = [m for m in TRACKED if self.sigmas.get(m) == 0]
        if deterministic:
            lines += ["", f"{len(deterministic)} metric(s) are perfectly stable - retrieval and "
                          "guard metrics", "should be, since no model sampling is involved."]
        return "\n".join(lines)


def measure(run_fn: Callable[[int], Any], repeats: int = 5,
            dataset: str = "golden_qa", sigma_multiple: float = 2.0,
            progress: Callable[[str], None] | None = None) -> NoiseFloor:
    """Run the same config `repeats` times and compute per-metric variance.

    `run_fn(i)` performs one full evaluation and returns a Report. Injected
    rather than called directly so this module stays testable without a model,
    and so a caller can vary the thread namespace per repeat.
    """
    if repeats < 2:
        raise ValueError("A noise floor needs at least 2 runs; 5 is the spec default.")

    summaries: list[dict] = []
    for i in range(repeats):
        if progress:
            progress(f"noise floor run {i + 1}/{repeats}")
        summaries.append(run_fn(i).summary)

    floor = NoiseFloor(n_repeats=repeats, dataset=dataset,
                       per_run=summaries, sigma_multiple=sigma_multiple)

    for metric in TRACKED:
        values = [s.get(metric) for s in summaries if s.get(metric) is not None]
        clean = [v for v in values if isinstance(v, (int, float)) and v == v]
        if len(clean) < 2:
            continue
        mu, sd = mean(clean), stdev(clean)
        floor.means[metric] = round(mu, 6)
        floor.sigmas[metric] = round(sd, 6)
        floor.mde[metric] = round(sd * sigma_multiple, 6)

    return floor


__all__ = ["measure", "NoiseFloor", "TRACKED"]
