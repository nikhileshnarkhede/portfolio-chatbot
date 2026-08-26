"""
run_logger.py - persist a turn as a JSON record.

The bridge between the app and `eval/`. Because `state.py` forbids computing
something and throwing it away, this module needs no instrumentation of its
own: it serializes fields that already exist.

Layout:

    runs/<experiment>/<run_id>/
        meta.json            when, which config, which index, how many turns
        resolved_config.json the exact config that ran, overrides included
        turns.jsonl          one JSON object per turn

`turns.jsonl` is append-only and flushed per turn, so a run that dies partway
through - rate limits, Ctrl-C - still leaves everything it completed. An eval
sweep that loses two hours of Groq quota to a crash is a bad afternoon.

Full chunk text is deliberately NOT stored. Retrieval metrics need to know
which chunks came back and in what order, not their contents; `chunk_refs`
gives ids and metadata, and the assembled context string is kept separately
under `observability.log_context` when you want to read it.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..state import EVAL_FIELDS, GraphState, chunk_refs

TURNS_FILE = "turns.jsonl"
META_FILE = "meta.json"


def turn_record(state: GraphState, cfg: AppConfig, **extra: Any) -> dict:
    """The serializable view of one turn."""
    obs = cfg.observability
    record: dict[str, Any] = {k: state.get(k) for k in EVAL_FIELDS}

    if obs.log_retrieved_chunk_ids:
        record["chunks"] = chunk_refs(state)
    if obs.log_context:
        record["context"] = state.get("context", "")
    if not obs.log_latency:
        record.pop("timings", None)
    else:
        record["latency_s"] = round(sum((state.get("timings") or {}).values()), 4)
    if not obs.log_token_usage:
        record.pop("token_usage", None)

    record.update(extra)
    return record


@dataclass
class RunLogger:
    """Writes one run's worth of turns. Use as a context manager."""

    cfg: AppConfig
    run_id: str
    directory: Path
    label: str = ""
    _handle: Any = None
    _count: int = 0
    _started: float = 0.0

    @classmethod
    def open(cls, cfg: AppConfig, label: str = "", run_id: str | None = None) -> "RunLogger":
        run_id = run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        directory = cfg.resolve(cfg.paths.runs_dir) / cfg.run.experiment_name / run_id
        directory.mkdir(parents=True, exist_ok=True)
        cfg.snapshot(directory)
        logger = cls(cfg=cfg, run_id=run_id, directory=directory, label=label)
        logger._handle = (directory / TURNS_FILE).open("w", encoding="utf-8")
        logger._started = time.perf_counter()
        return logger

    def log(self, state: GraphState, **extra: Any) -> dict:
        record = turn_record(state, self.cfg, **extra)
        if self.cfg.observability.log_runs and self._handle:
            self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._handle.flush()  # survive a crash mid-sweep
        self._count += 1
        return record

    def close(self, **extra: Any) -> Path:
        meta = {
            "run_id": self.run_id,
            "label": self.label,
            "experiment": self.cfg.run.experiment_name,
            "run_fingerprint": self.cfg.run_fingerprint,
            "index_fingerprint": self.cfg.index_fingerprint,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.perf_counter() - self._started, 2),
            "n_turns": self._count,
            **extra,
        }
        (self.directory / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self._handle:
            self._handle.close()
            self._handle = None
        return self.directory

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        if self._handle:
            self.close()


def load_turns(run_dir: Path) -> list[dict]:
    """Read a run's turns back. Tolerates a truncated final line from a crash."""
    path = Path(run_dir) / TURNS_FILE
    if not path.exists():
        raise FileNotFoundError(f"No {TURNS_FILE} in {run_dir}")
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            break  # partial write from an interrupted run
    return turns


def load_meta(run_dir: Path) -> dict:
    path = Path(run_dir) / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def list_runs(cfg: AppConfig, experiment: str | None = None) -> list[Path]:
    """Run directories, newest first."""
    root = cfg.resolve(cfg.paths.runs_dir)
    base = root / experiment if experiment else root
    if not base.exists():
        return []
    runs = [p for p in base.rglob(TURNS_FILE)]
    return sorted((p.parent for p in runs), key=lambda p: p.name, reverse=True)


def latest_run(cfg: AppConfig, experiment: str | None = None) -> Path | None:
    runs = list_runs(cfg, experiment)
    return runs[0] if runs else None


__all__ = [
    "RunLogger", "turn_record", "load_turns", "load_meta",
    "list_runs", "latest_run", "TURNS_FILE", "META_FILE",
]
