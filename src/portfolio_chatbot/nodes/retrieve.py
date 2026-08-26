"""
retrieve.py - run the search and assemble the context block.

Thin by design: `tools/retriever_tool.py` does the work so it can be tested
without a graph. This node's job is to move data between the state and that
tool, and to record what actually happened.

Ranking uses the EXPANDED question; the chunk_type filter comes from the route
decision, which was made on the RAW question. Keeping those two inputs
separate is the whole reason both fields exist in the state.
"""

from __future__ import annotations

import time
from langchain_core.runnables import RunnableConfig
from ..state import GraphState
from ..tools import retriever_tool
from . import app_config


def retrieve(state: GraphState, config: RunnableConfig) -> dict:
    cfg = app_config(config)
    started = time.perf_counter()

    search_text = state.get("expanded_question") or state.get("question", "")
    route = state.get("route") or {}
    chunk_types = list(route.get("chunk_types") or [])

    docs, mode = retriever_tool.search(cfg, search_text, chunk_types)
    context, truncated = retriever_tool.format_context(docs, cfg)

    note = f"{len(docs)} docs via {mode}"
    if truncated:
        note += f", context clipped at {cfg.retrieval.context.max_context_chars}"

    return {
        "documents": docs,
        "retrieval_mode": mode,
        "context": context,
        "context_chars": len(context),
        "context_truncated": truncated,
        "timings": {"retrieve": round(time.perf_counter() - started, 4)},
        "trace": [f"retrieve: {note}"],
    }


__all__ = ["retrieve"]
