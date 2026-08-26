"""
gate.py - turn a report into a pass/fail decision.

`compare.py` renders deltas and expects a human to read them. This decides, so
a regression can stop a deploy instead of sitting in a log nobody opened.

Three independent checks, in order of severity:

**Invariants** hold absolutely. `leaked_url_count` must be zero: a forged link
reaching a recruiter is not a metric that got worse, it is a broken guard.
These fire on the very first run, before any baseline exists, and no tolerance
applies.

**Thresholds** are absolute bars from `eval/thresholds.json` - the file you own.
Each names its own `on_fail`, so promoting a warning to a blocker is a one-word
edit. `faithfulness` additionally carries `requires_kappa`: it cannot block a
release until the judge has been validated to that agreement level, so an
unvalidated instrument never gets to stop you shipping.

**Regression** compares against the recorded baseline in sigma units. This is
the check that catches creep - a config can satisfy every absolute threshold
while being measurably worse than what it replaced.

    python scripts/run_eval.py --repeats 5 --save-baseline   # measure the floor
    python scripts/run_eval.py --assert                      # hold the line
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portfolio_chatbot.config import PROJECT_ROOT

THRESHOLDS_PATH = PROJECT_ROOT / "eval" / "thresholds.json"
BASELINE_DIR = PROJECT_ROOT / "eval" / "baselines"


# ==========================================================================

@dataclass
class Violation:
    kind: str          # "invariant" | "threshold" | "regression"
    severity: str      # "block" | "warn"
    metric: str
    expected: str
    actual: float
    message: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class GateResult:
    passed: bool = True                 # no blocking violations
    clean: bool = True                  # no violations of any kind
    violations: list[Violation] = field(default_factory=list)
    checked: int = 0
    skipped: list[str] = field(default_factory=list)
    baseline_used: str | None = None

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "block"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warn"]

    def as_dict(self) -> dict:
        return {
            "passed": self.passed, "clean": self.clean, "checked": self.checked,
            "skipped": self.skipped, "baseline_used": self.baseline_used,
            "violations": [v.as_dict() for v in self.violations],
        }

    def render(self) -> str:
        head = "GATE PASS" if self.passed else "GATE FAIL"
        detail = f"{self.checked} checked"
        if self.baseline_used:
            detail += f", baseline {self.baseline_used}"
        if self.warnings:
            detail += f", {len(self.warnings)} warning(s)"
        lines = [f"{head}  ({detail})"]

        for group, label in ((self.blocking, "BLOCK"), (self.warnings, "warn")):
            for v in group:
                lines += ["", f"  [{label}] {v.metric}  ({v.kind})",
                          f"      {v.message}"]

        if self.skipped:
            lines += ["", "  not checked: " + ", ".join(self.skipped)]

        if not self.passed:
            lines += ["", "If this change is intended, re-record the baseline:",
                      "    python scripts/run_eval.py --repeats 5 --save-baseline"]
        return "\n".join(lines)


# ==========================================================================

def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No thresholds file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_path(dataset: str) -> Path:
    return BASELINE_DIR / f"{dataset}.json"


def load_baseline(dataset: str) -> dict | None:
    path = baseline_path(dataset)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_baseline(report: Any, sigmas: dict[str, float] | None = None,
                  n_repeats: int = 1) -> Path:
    """Record a report's metrics, and their spread, as the bar to hold.

    `sigmas` is what makes the regression check meaningful. Without it every
    tolerance falls back to `min_absolute`, which is a guess rather than a
    measurement of how much this system actually wobbles.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path(report.dataset)
    path.write_text(json.dumps({
        "_meta": {
            "description": "Regression baseline. Re-record deliberately after an "
                           "intended improvement - never to silence a failure you "
                           "have not understood.",
            "n_repeats": n_repeats,
            "sigma_measured": bool(sigmas),
        },
        "dataset": report.dataset,
        "experiment": report.experiment,
        "run_fingerprint": report.run_fingerprint,
        "index_fingerprint": report.index_fingerprint,
        "n_turns": report.n_turns,
        "summary": _numeric_only(report.summary),
        "sigma": sigmas or {},
    }, indent=2, default=str), encoding="utf-8")
    return path


def _numeric_only(summary: dict) -> dict:
    out = {}
    for k, v in summary.items():
        n = _num(v)
        if n is not None:
            out[k] = n
    return out


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


# ==========================================================================

def check_invariants(summary: dict, spec: dict) -> list[Violation]:
    violations = []
    for metric, rule in (spec.get("invariants") or {}).items():
        actual = _num(summary.get(metric))
        if actual is None:
            continue  # not measured by this suite; not a pass and not a failure
        why = rule.get("why", "")
        if "max" in rule and actual > float(rule["max"]):
            violations.append(Violation(
                "invariant", "block", metric, f"<= {rule['max']}", actual,
                f"{actual:g} exceeds the hard limit of {rule['max']}. {why}",
            ))
        if "min" in rule and actual < float(rule["min"]):
            violations.append(Violation(
                "invariant", "block", metric, f">= {rule['min']}", actual,
                f"{actual:g} is below the hard floor of {rule['min']}. {why}",
            ))
    return violations


def check_thresholds(summary: dict, spec: dict, kappa: float | None = None) -> tuple[list[Violation], int, list[str]]:
    violations: list[Violation] = []
    checked = 0
    skipped: list[str] = []

    for metric, rule in (spec.get("thresholds") or {}).items():
        actual = _num(summary.get(metric))
        if actual is None:
            skipped.append(metric)
            continue

        severity = rule.get("on_fail", "warn")

        # A judged metric may not block until the judge has earned it.
        required_kappa = rule.get("requires_kappa")
        if required_kappa is not None and severity == "block":
            if kappa is None or kappa < float(required_kappa):
                severity = "warn"

        checked += 1
        if "min" in rule and actual < float(rule["min"]):
            violations.append(Violation(
                "threshold", severity, metric, f">= {rule['min']}", actual,
                f"{actual:.4g} is below the required {rule['min']}"
                + ("" if severity == "block" else
                   f" (demoted to warn: judge kappa {kappa if kappa is not None else 'unmeasured'}"
                   f" < {required_kappa})" if rule.get("requires_kappa") else ""),
            ))
        if "max" in rule and actual > float(rule["max"]):
            violations.append(Violation(
                "threshold", severity, metric, f"<= {rule['max']}", actual,
                f"{actual:.4g} exceeds the limit of {rule['max']}",
            ))

    return violations, checked, skipped


def check_regression(summary: dict, spec: dict, baseline: dict) -> tuple[list[Violation], int]:
    """Compare against the baseline in sigma units.

    A metric may drift by `sigma_multiple` standard deviations before it counts
    as a regression - so the gate fires on real movement rather than on the
    noise the system produces when nothing changes at all.
    """
    rules = spec.get("regression") or {}
    multiple = float(rules.get("sigma_multiple", 2.0))
    floor = float(rules.get("min_absolute", 0.02))

    old = baseline.get("summary", {})
    sigmas = baseline.get("sigma", {})
    thresholds = spec.get("thresholds") or {}

    violations: list[Violation] = []
    checked = 0

    for metric, rule in thresholds.items():
        before, after = _num(old.get(metric)), _num(summary.get(metric))
        if before is None or after is None:
            continue
        checked += 1

        sigma = _num(sigmas.get(metric))
        tolerance = max(sigma * multiple, floor) if sigma is not None else floor
        higher_is_better = "min" in rule
        drop = (before - after) if higher_is_better else (after - before)

        if drop > tolerance:
            basis = (f"{multiple}sigma = {tolerance:.4g}" if sigma is not None
                     else f"{tolerance:.4g} (no measured sigma - run --repeats to fix)")
            violations.append(Violation(
                "regression", rule.get("on_fail", "warn"), metric,
                f"within {basis} of {before:.4g}", after,
                f"{before:.4g} -> {after:.4g} ({after - before:+.4g}), tolerance {basis}",
            ))

    return violations, checked


def check(report: Any, baseline: dict | None = None, spec: dict | None = None,
          kappa: float | None = None) -> GateResult:
    """Run every applicable check against a report."""
    spec = spec or load_thresholds()
    summary = report.summary

    violations = check_invariants(summary, spec)
    threshold_violations, checked, skipped = check_thresholds(summary, spec, kappa)
    violations += threshold_violations

    baseline_name = None
    if baseline:
        baseline_name = f"{baseline.get('dataset')}@{baseline.get('run_fingerprint', '')[:8]}"
        regression_violations, regression_checked = check_regression(summary, spec, baseline)
        violations += regression_violations
        checked += regression_checked

    blocking = [v for v in violations if v.severity == "block"]
    return GateResult(
        passed=not blocking, clean=not violations, violations=violations,
        checked=checked, skipped=sorted(skipped), baseline_used=baseline_name,
    )


__all__ = [
    "check", "check_invariants", "check_thresholds", "check_regression",
    "load_thresholds", "load_baseline", "save_baseline", "baseline_path",
    "GateResult", "Violation", "THRESHOLDS_PATH", "BASELINE_DIR",
]
