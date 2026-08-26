"""
perf.py - process-level timing registry for one-time costs.

Per-turn timings live in the graph state, where they belong. This module exists
for the costs that happen *once per process* and are therefore invisible to any
per-turn measurement: loading the FAISS index, initialising the embedding model.

Those are the cold-start numbers. On Streamlit Community Cloud a container that
has gone to sleep pays them again on the next visitor, so "how long before the
first answer" is not `latency_p50` - it is `cold_start + latency_p50`, and only
one of those two terms shows up in a turn-level report.

Deliberately a module-level registry rather than something threaded through the
call graph: the cost is incurred deep inside a cached factory, several frames
below anything that knows what a "run" is, and plumbing a timing object down
there would distort the code it measures.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

#: name -> seconds, first observation wins.
_TIMINGS: dict[str, float] = {}


def record(name: str, seconds: float) -> None:
    """Record a one-time cost. First write wins.

    First-write-wins is the point: these measure *cold* start. A second call is
    by definition a warm path, and overwriting would replace the number you
    care about with one you don't.
    """
    _TIMINGS.setdefault(name, round(seconds, 4))


@contextmanager
def timed(name: str):
    """Time a block and record it under `name`."""
    started = time.perf_counter()
    try:
        yield
    finally:
        record(name, time.perf_counter() - started)


def snapshot() -> dict[str, float]:
    """Everything recorded so far, plus the total."""
    out = dict(_TIMINGS)
    if out:
        out["cold_start_total"] = round(sum(_TIMINGS.values()), 4)
    return out


def get(name: str) -> float | None:
    return _TIMINGS.get(name)


def reset() -> None:
    """Clear the registry. Used by tests, and by a deliberate re-measure."""
    _TIMINGS.clear()


__all__ = ["record", "timed", "snapshot", "get", "reset"]
