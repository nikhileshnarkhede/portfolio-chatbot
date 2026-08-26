"""
calibration.py - does the judge agree with a human?

Spec §06. Without this, "faithfulness = 0.94" is a number produced by an
unvalidated instrument, and optimising against it is optimising against noise.

Cohen's kappa is used rather than raw agreement because raw agreement is
inflated by class imbalance: if 90% of the calibration cases are supported, a
judge that answers "supported" every single time scores 90% agreement while
carrying no information at all. Kappa corrects for exactly that, which is why
`eval/datasets/judge_calibration.json` is deliberately balanced near 50/50.

The verdict feeds straight into the gate. Below `min_kappa_direction` the judged
column is suppressed entirely rather than shown with a caveat nobody reads;
between the thresholds it is direction-only; at or above `min_kappa_trust` it
may block a release.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from portfolio_chatbot.config import PROJECT_ROOT, AppConfig

DATASET = PROJECT_ROOT / "eval" / "datasets" / "judge_calibration.json"
RESULT_PATH = PROJECT_ROOT / "eval" / "baselines" / "judge_calibration.json"

NAN = float("nan")


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Chance-corrected agreement between two binary raters.

    Returns NaN for fewer than two pairs. Returns 1.0 when both raters are
    constant AND identical - degenerate but genuinely perfect agreement; and
    0.0 when they are constant and opposed.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return NAN

    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n

    a_true = sum(1 for x, _ in pairs if x) / n
    b_true = sum(1 for _, y in pairs if y) / n
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)

    if expected >= 1.0:
        # Both raters constant. Perfect agreement is real; disagreement is not
        # possible to chance-correct, so report the raw outcome.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


@dataclass
class CalibrationResult:
    kappa: float = NAN
    agreement: float = NAN
    n: int = 0
    n_unlabelled: int = 0
    n_judge_failed: int = 0
    verdict: str = "unmeasured"      # trust | direction | suppress | unmeasured
    label_source: str = "expected"   # human | expected | mixed
    disagreements: list[dict] = field(default_factory=list)
    judge_model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @property
    def may_block(self) -> bool:
        return self.verdict == "trust"

    def render(self) -> str:
        lines = [
            f"JUDGE CALIBRATION   {self.judge_model}",
            "",
            f"  cases scored     {self.n}"
            + (f"   ({self.n_judge_failed} judge failures)" if self.n_judge_failed else ""),
            f"  raw agreement    {self.agreement:.3f}" if self.agreement == self.agreement else "  raw agreement    n/a",
            f"  Cohen's kappa    {self.kappa:.3f}" if self.kappa == self.kappa else "  Cohen's kappa    n/a",
            f"  labels from      {self.label_source}",
            "",
        ]
        verdicts = {
            "trust": "TRUST - the judged column may block a release.",
            "direction": "DIRECTION ONLY - report the number, never gate on it.",
            "suppress": "SUPPRESS - agreement is too low for the column to mean anything.",
            "unmeasured": "UNMEASURED - run calibration before reading any judged metric.",
        }
        lines.append(f"  verdict: {verdicts[self.verdict]}")

        if self.label_source == "expected":
            lines += ["", "  NOTE: scored against authored labels, not your own grading.",
                      "  Seeded fabrications are objectively ungrounded so this is",
                      "  meaningful, but human labels are stronger evidence."]
        if self.disagreements:
            lines += ["", f"  disagreements ({len(self.disagreements)}):"]
            for d in self.disagreements[:8]:
                lines.append(f"    {d['id']:<28} human={d['human']}  judge={d['judge']}")
        return "\n".join(lines)


def load_cases(path: Path = DATASET) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No calibration dataset at {path}")
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def human_label(case: dict) -> bool | None:
    """Your grading if present, else the authored label.

    `human_label` is null until graded in the dashboard. `expected_label` is the
    authored intent, and for the seeded-fabrication half it is objective - the
    claim either does or does not appear in the context.
    """
    if case.get("human_label") is not None:
        return bool(case["human_label"])
    if case.get("expected_label") is not None:
        return bool(case["expected_label"])
    return None


def calibrate(cfg: AppConfig, judge: Any, cases: list[dict] | None = None,
              progress: Callable[[str], None] | None = None) -> CalibrationResult:
    """Score the calibration set with `judge` and compare to the human labels."""
    cases = cases if cases is not None else load_cases()

    human: list[bool] = []
    machine: list[bool] = []
    disagreements: list[dict] = []
    unlabelled = 0
    judge_failed = 0
    sources: set[str] = set()

    for case in cases:
        truth = human_label(case)
        if truth is None:
            unlabelled += 1
            continue
        sources.add("human" if case.get("human_label") is not None else "expected")

        if progress:
            progress(f"calibrating {case['id']}")
        verdict = judge.supports(case["context"], case["claim"])
        if verdict is None:
            judge_failed += 1
            continue

        human.append(truth)
        machine.append(bool(verdict))
        if truth != bool(verdict):
            disagreements.append({
                "id": case["id"], "human": truth, "judge": bool(verdict),
                "claim": case["claim"][:120],
            })

    kappa = cohens_kappa(human, machine)
    agreement = (sum(1 for x, y in zip(human, machine) if x == y) / len(human)
                 if human else NAN)

    calib = cfg.eval.calibration
    if kappa != kappa:
        verdict = "unmeasured"
    elif kappa >= calib.min_kappa_trust:
        verdict = "trust"
    elif kappa >= calib.min_kappa_direction:
        verdict = "direction"
    else:
        verdict = "suppress"

    return CalibrationResult(
        kappa=kappa, agreement=agreement, n=len(human),
        n_unlabelled=unlabelled, n_judge_failed=judge_failed,
        verdict=verdict,
        label_source="mixed" if len(sources) > 1 else (sources.pop() if sources else "expected"),
        disagreements=disagreements,
        judge_model=getattr(judge, "model", "") or "",
    )


def save(result: CalibrationResult, path: Path = RESULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load() -> CalibrationResult | None:
    if not RESULT_PATH.exists():
        return None
    data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    return CalibrationResult(**data)


def current_kappa() -> float | None:
    """Kappa from the last calibration, for the gate. None if never measured."""
    result = load()
    if result is None or result.kappa != result.kappa:
        return None
    return result.kappa


__all__ = [
    "cohens_kappa", "calibrate", "CalibrationResult", "load_cases",
    "human_label", "save", "load", "current_kappa", "DATASET", "RESULT_PATH",
]
