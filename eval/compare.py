"""
compare.py - put two runs side by side.

A single report tells you what a config scored. It cannot tell you whether a
change helped, which is the only question that matters. This renders the delta,
and flags the per-case regressions that an aggregate average hides: a chunking
change that lifts mean recall while breaking the two questions you most care
about looks like an improvement in the summary row.

    python scripts/run_eval.py --compare exp001_baseline exp002_chunk_512

A word on reading the deltas. These are means over ~22 cases against a
non-deterministic model on a free tier. Treat small movements as noise. A
change worth acting on should be visible in the per-case table, not only in the
third decimal place of an average.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Metrics where a higher number is better. Everything else is inverted when
#: deciding whether a delta is an improvement.
HIGHER_IS_BETTER = {
    "route_correct", "hit_at_k", "recall_at_k", "mrr", "type_precision",
    "fact_coverage", "context_overlap", "first_person", "refusal_correct",
    "faithfulness", "relevancy",
}
LOWER_IS_BETTER = {
    "attempted_forged_url", "forged_url_count_total", "leaked_url_count",
    "context_truncated_count", "n_errors", "latency_mean",
}
HEADLINE = [
    "route_correct", "hit_at_k", "recall_at_k", "mrr", "type_precision",
    "fact_coverage", "context_overlap", "first_person",
    "attempted_forged_url", "refusal_correct", "leaked_url_count",
    "context_truncated_count", "latency_mean", "n_errors",
]


def load_report(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Report not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def find_report(reports_dir: Path, experiment: str, dataset: str = "golden_qa") -> Path:
    """Newest report for an experiment+dataset pair."""
    matches = sorted(reports_dir.glob(f"{experiment}__{dataset}__*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(
            f"No report for '{experiment}' on '{dataset}' in {reports_dir}. "
            f"Run: python scripts/run_eval.py --experiment {experiment}"
        )
    return matches[0]


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _verdict(metric: str, delta: float) -> str:
    if abs(delta) < 1e-9:
        return "="
    better = delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
    if metric not in HIGHER_IS_BETTER and metric not in LOWER_IS_BETTER:
        return "?"
    return "better" if better else "WORSE"


def compare(baseline: dict, candidate: dict) -> dict:
    """Metric deltas plus per-case regressions."""
    rows = []
    for metric in HEADLINE:
        a = _num(baseline["summary"].get(metric))
        b = _num(candidate["summary"].get(metric))
        if a is None and b is None:
            continue
        delta = None if (a is None or b is None) else b - a
        rows.append({
            "metric": metric,
            "baseline": a,
            "candidate": b,
            "delta": delta,
            "verdict": _verdict(metric, delta) if delta is not None else "n/a",
        })

    return {
        "baseline": baseline["experiment"],
        "candidate": candidate["experiment"],
        "dataset": baseline.get("dataset"),
        "metrics": rows,
        "case_changes": case_changes(baseline, candidate),
    }


def case_changes(baseline: dict, candidate: dict,
                 metrics: tuple[str, ...] = ("hit_at_k", "recall_at_k", "route_correct")) -> list[dict]:
    """Per-case movements. This is where an averaged win hides a real loss."""
    def key(turn: dict) -> str:
        return f"{turn['case_id']}[{turn['turn_index']}]"

    before = {key(t): t for t in baseline["turns"]}
    changes = []

    for turn in candidate["turns"]:
        k = key(turn)
        old = before.get(k)
        if not old:
            continue
        for metric in metrics:
            a = _num((old.get("retrieval") or {}).get(metric))
            b = _num((turn.get("retrieval") or {}).get(metric))
            if a is None or b is None or abs(b - a) < 1e-9:
                continue
            changes.append({
                "case": k, "metric": metric, "baseline": a, "candidate": b,
                "delta": b - a, "verdict": _verdict(metric, b - a),
                "question": turn.get("question", "")[:70],
            })

    changes.sort(key=lambda c: (c["verdict"] != "WORSE", -abs(c["delta"])))
    return changes


def render(result: dict, max_cases: int = 15) -> str:
    lines = [
        f"{result['baseline']}  ->  {result['candidate']}   [{result['dataset']}]",
        "",
        f"{'metric':<24}{'baseline':>10}{'candidate':>11}{'delta':>10}   verdict",
        "-" * 68,
    ]
    for row in result["metrics"]:
        a = "n/a" if row["baseline"] is None else f"{row['baseline']:.3f}"
        b = "n/a" if row["candidate"] is None else f"{row['candidate']:.3f}"
        d = "" if row["delta"] is None else f"{row['delta']:+.3f}"
        lines.append(f"{row['metric']:<24}{a:>10}{b:>11}{d:>10}   {row['verdict']}")

    changes = result["case_changes"]
    regressions = [c for c in changes if c["verdict"] == "WORSE"]

    lines += ["", f"per-case changes: {len(changes)}  ({len(regressions)} regressions)"]
    if changes:
        lines.append("-" * 68)
        for c in changes[:max_cases]:
            lines.append(
                f"  {c['verdict']:<7} {c['case']:<28} {c['metric']:<14} "
                f"{c['baseline']:.2f} -> {c['candidate']:.2f}"
            )
        if len(changes) > max_cases:
            lines.append(f"  ... {len(changes) - max_cases} more")

    if regressions:
        lines += ["", "Regressions above are individual questions that got worse. "
                      "Check them before accepting an improved average."]
    return "\n".join(lines)


__all__ = ["compare", "render", "load_report", "find_report", "case_changes"]
