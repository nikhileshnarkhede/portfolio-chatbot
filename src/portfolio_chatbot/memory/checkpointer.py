"""
checkpointer.py - conversation persistence for the graph.

`memory.checkpointer` selects the backend:

``memory``  per-process, gone on restart. Right for Streamlit (a session lives
            in one process anyway) and for eval runs.
``sqlite``  survives restarts. Needed if you want a conversation to resume
            after a redeploy, or to inspect threads after the fact.
``none``    no threading at all. Each invocation is independent - which is what
            you want when every eval case must be isolated and history from
            case 2 must not reach case 3.

Isolated here rather than left inline in `graph.py` so that swapping backends
is a one-file change, and so `graph.py` keeps to topology.
"""

from __future__ import annotations

from typing import Any

from ..config import AppConfig

CHECKPOINT_DB = "checkpoints.sqlite"


def build_checkpointer(cfg: AppConfig) -> Any | None:
    """The checkpointer named by `memory.checkpointer`, or None for stateless."""
    kind = cfg.memory.checkpointer

    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if kind == "sqlite":  # pragma: no cover - optional dependency
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise ImportError(
                "memory.checkpointer='sqlite' needs langgraph-checkpoint-sqlite:\n"
                "    pip install langgraph-checkpoint-sqlite\n"
                "Or set memory.checkpointer to 'memory'."
            ) from exc

        path = cfg.resolve(cfg.paths.runs_dir) / CHECKPOINT_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver.from_conn_string(str(path))

    raise ValueError(  # pragma: no cover - config validation catches this first
        f"Unknown checkpointer: {kind!r}"
    )


__all__ = ["build_checkpointer", "CHECKPOINT_DB"]
