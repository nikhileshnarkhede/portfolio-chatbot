"""
apply.py - promote a tuned config to the live one (spec §10).

One rule governs this module: **you cannot apply a config you have not
measured.** Apply is refused unless a passing evaluation exists for that exact
`run_fingerprint`, which is what stops the dashboard from becoming a set of
sliders that change production on a hunch.

Per your decision, Apply **overwrites** `configs/default.json`. One file, no
indirection: what the app reads is what you tuned. That has a real cost and
three mitigations.

The cost: experiment files are *deltas over* `default.json`. Overwrite it and an
experiment that said `max_chunk_chars: 512` becomes a no-op the moment default
is 512.

The mitigations, in order of importance:

1. **Past results are untouched.** Every run already snapshots its own fully
   resolved config to `runs/.../resolved_config.json`, so a six-month-old report
   can still be read exactly, whatever default.json says now.
2. **The replaced config is archived** to `configs/history/` before the write,
   so rollback is a file copy rather than a re-tune.
3. **Deltas are re-resolved on apply** and any that became a no-op is reported,
   rather than left to rot.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portfolio_chatbot.config import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    EXPERIMENT_DIR,
    PROJECT_ROOT,
    deep_merge,
)

HISTORY_DIR = CONFIG_DIR / "history"
PROVENANCE_PATH = CONFIG_DIR / "provenance.json"


class ApplyRefused(RuntimeError):
    """Apply was blocked. The message says why."""


@dataclass
class ApplyResult:
    experiment_name: str
    run_fingerprint: str
    archived_to: str
    experiment_file: str
    stale_experiments: list[str] = field(default_factory=list)
    overridden: bool = False
    note: str = ""

    def render(self) -> str:
        lines = [
            f"APPLIED   {self.experiment_name}",
            f"  run fingerprint  {self.run_fingerprint}",
            f"  previous config  archived to {self.archived_to}",
            f"  delta recorded   {self.experiment_file}",
        ]
        if self.overridden:
            lines += ["", f"  OVERRIDE USED: {self.note}"]
        if self.stale_experiments:
            lines += ["", "  These experiment files are now no-ops against the new",
                      "  default and no longer describe a change:",
                      "    " + ", ".join(self.stale_experiments)]
        return "\n".join(lines)


# ==========================================================================

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "config"


def next_experiment_name(label: str) -> str:
    """expNNN_<slug>, numbered after whatever already exists."""
    existing = sorted(EXPERIMENT_DIR.glob("exp*.json"))
    highest = 0
    for path in existing:
        m = re.match(r"exp(\d+)", path.stem)
        if m:
            highest = max(highest, int(m.group(1)))
    return f"exp{highest + 1:03d}_{_slug(label)}"


def compute_delta(new_config: dict, base: dict) -> dict:
    """Only what differs, recursively. Empty means nothing changed."""
    delta: dict[str, Any] = {}
    for key, value in new_config.items():
        if key.startswith("_"):
            continue
        if key not in base:
            delta[key] = value
        elif isinstance(value, dict) and isinstance(base[key], dict):
            nested = compute_delta(value, base[key])
            if nested:
                delta[key] = nested
        elif value != base[key]:
            delta[key] = value
    return delta


def stale_experiments(new_default: dict) -> list[str]:
    """Experiment files that no longer describe a change against the new default."""
    stale = []
    for path in sorted(EXPERIMENT_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        delta = {k: v for k, v in raw.items() if not k.startswith("_") and k != "run"}
        if not delta:
            continue
        if deep_merge(new_default, delta) == new_default:
            stale.append(path.stem)
    return stale


def check_ready(report: Any, gate_result: Any, cfg: Any) -> list[str]:
    """Everything blocking an apply. Empty list means go."""
    problems: list[str] = []

    if report is None:
        problems.append("No evaluation has been run for this config.")
    elif getattr(report, "run_fingerprint", None) != cfg.run_fingerprint:
        problems.append(
            f"The evaluation was run against a different config "
            f"({getattr(report, 'run_fingerprint', '?')[:8]}, this one is "
            f"{cfg.run_fingerprint[:8]}). Re-run before applying."
        )

    if not cfg.index_path.exists():
        problems.append(f"No index exists for {cfg.index_fingerprint}. Run ingest first.")

    if gate_result is not None and not getattr(gate_result, "passed", True):
        blocking = ", ".join(v.metric for v in gate_result.blocking)
        problems.append(f"The gate failed on: {blocking}")

    return problems


def apply(cfg: Any, new_config: dict, report: Any, gate_result: Any,
          label: str = "", override_note: str = "") -> ApplyResult:
    """Promote `new_config` to the live default.

    `override_note` is the escape hatch for a deliberately-accepted gate
    failure. It requires words, not a checkbox, and it is recorded in the
    provenance file - a decision you have to write down is one you have to
    actually make.
    """
    problems = check_ready(report, gate_result, cfg)
    overriding = bool(problems) and bool(override_note.strip())

    if problems and not overriding:
        raise ApplyRefused(
            "Apply refused:\n  - " + "\n  - ".join(problems)
            + "\n\nFix these, or supply an override note explaining why they are acceptable."
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    archive = HISTORY_DIR / f"default_{stamp}_{cfg.run_fingerprint[:8]}.json"
    archive.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")

    delta = compute_delta(new_config, base)
    name = next_experiment_name(label or "tuned")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_path = EXPERIMENT_DIR / f"{name}.json"
    experiment_path.write_text(json.dumps({
        "_meta": {
            "description": f"Applied to the live config on {stamp}.",
            "applied": True,
            "run_fingerprint": cfg.run_fingerprint,
            "changes": sorted(_flatten(delta)),
        },
        **delta,
        "run": {"experiment_name": name, "notes": label or "applied from dashboard"},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Preserve the _meta block: it carries the rules that keep this file honest.
    merged = dict(new_config)
    if "_meta" in base:
        merged["_meta"] = base["_meta"]
    DEFAULT_CONFIG_PATH.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    result = ApplyResult(
        experiment_name=name,
        run_fingerprint=cfg.run_fingerprint,
        archived_to=str(archive.relative_to(PROJECT_ROOT)),
        experiment_file=str(experiment_path.relative_to(PROJECT_ROOT)),
        stale_experiments=stale_experiments(merged),
        overridden=overriding,
        note=override_note,
    )
    _record_provenance(cfg, report, gate_result, result, stamp)
    return result


def _flatten(node: dict, prefix: str = "") -> list[str]:
    out = []
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_flatten(value, path))
        else:
            out.append(f"{path}={value}")
    return out


def _record_provenance(cfg: Any, report: Any, gate_result: Any,
                       result: ApplyResult, stamp: str) -> None:
    """Why this config is live. Answers 'what justified this?' months later."""
    history = (json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
               if PROVENANCE_PATH.exists() else {"applied": []})
    history.setdefault("_meta", {
        "description": "Append-only record of every config promoted to live, and the "
                       "evaluation that justified it.",
    })
    history["applied"].append({
        "applied_at": stamp,
        "experiment": result.experiment_name,
        "run_fingerprint": cfg.run_fingerprint,
        "index_fingerprint": cfg.index_fingerprint,
        "archived_previous": result.archived_to,
        "report": {
            "dataset": getattr(report, "dataset", None),
            "n_turns": getattr(report, "n_turns", None),
            "summary": {k: v for k, v in (getattr(report, "summary", {}) or {}).items()
                        if isinstance(v, (int, float))},
        } if report is not None else None,
        "gate": gate_result.as_dict() if hasattr(gate_result, "as_dict") else None,
        "override_note": result.note or None,
        "stale_experiments": result.stale_experiments,
    })
    PROVENANCE_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                               encoding="utf-8")


def list_history() -> list[Path]:
    """Archived configs, newest first."""
    if not HISTORY_DIR.exists():
        return []
    return sorted(HISTORY_DIR.glob("default_*.json"), reverse=True)


def rollback(archive: Path) -> Path:
    """Restore an archived config, archiving the current one first."""
    archive = Path(archive)
    if not archive.exists():
        raise FileNotFoundError(f"No archived config at {archive}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    (HISTORY_DIR / f"default_{stamp}_prerollback.json").write_text(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    DEFAULT_CONFIG_PATH.write_text(archive.read_text(encoding="utf-8"), encoding="utf-8")
    return DEFAULT_CONFIG_PATH


def load_provenance() -> list[dict]:
    if not PROVENANCE_PATH.exists():
        return []
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")).get("applied", [])


__all__ = [
    "apply", "check_ready", "compute_delta", "stale_experiments",
    "next_experiment_name", "rollback", "list_history", "load_provenance",
    "ApplyResult", "ApplyRefused", "HISTORY_DIR", "PROVENANCE_PATH",
]
