"""
route.py - question -> chunk_type filter.

Ports the `TYPE_ROUTES` table from app.py, now living in
`retrieval.routing.routes`. Rules are evaluated in config order and the first
keyword hit wins, so ordering is meaningful: "current job" must be checked
before the generic "job", or every question about a current role would be
answered from the whole employment history.

Routing runs on the RAW question, never the expanded one. That was a deliberate
choice in the original and it still holds: query expansion appends words like
"projects applications systems", which would make an expanded question about,
say, education match the projects rule instead.
`retrieval.routing.route_on` exposes the choice so the alternative can be
measured rather than argued about.

Which rule fired is recorded even when nothing matched. The share of questions
falling through to MMR is a metric in its own right - it is how you notice a
routing-table edit quietly changing retrieval for questions you never tested.
"""

from __future__ import annotations

import time
from langchain_core.runnables import RunnableConfig
from ..state import GraphState, RouteDecision
from . import app_config


def _decision(matched: bool, rule=None, keyword=None, types=None) -> RouteDecision:
    return RouteDecision(
        matched=matched, rule_name=rule, matched_keyword=keyword,
        chunk_types=list(types or []),
    )


def route(state: GraphState, config: RunnableConfig) -> dict:
    cfg = app_config(config)
    started = time.perf_counter()
    routing = cfg.retrieval.routing

    def done(decision: RouteDecision, note: str) -> dict:
        return {
            "route": decision,
            "timings": {"route": round(time.perf_counter() - started, 4)},
            "trace": [f"route: {note}"],
        }

    if not routing.enabled:
        return done(_decision(False), "disabled")

    source = (state.get("expanded_question") if routing.route_on == "expanded_question"
              else state.get("question")) or ""
    text = source.lower()

    for rule in routing.routes:
        for keyword in rule.keywords:
            if keyword in text:
                return done(
                    _decision(True, rule.name, keyword, rule.types),
                    f"{rule.name} (on '{keyword}') -> {list(rule.types)}",
                )

    return done(_decision(False), "no match -> mmr")


__all__ = ["route"]
