"""
sanitize.py - enforce the URL allowlist on the model's output.

Reads `draft_answer`, writes `answer` plus a `link_audit`. Both survive into
the state, which is what makes URL hallucination measurable: `draft_answer`
records what the model tried to emit, `answer` is what a user would have seen.

This node runs unconditionally, after every generation, including when
generation failed and `draft_answer` is an error message. A guard that only
runs on the happy path is not a guard.

It is also where the reply enters conversation history, which is a deliberate
placement rather than a convenience. The answer is not final until the guard
has run, and appending `draft_answer` to `messages` would put a forged URL into
the history that every later turn is conditioned on - the model would then see
its own hallucinated link as an established fact of the conversation.
"""

from __future__ import annotations

import time
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..state import GraphState
from ..tools.link_guard import sanitize as strip_links
from . import app_config


def sanitize(state: GraphState, config: RunnableConfig) -> dict:
    cfg = app_config(config)
    started = time.perf_counter()

    draft = state.get("draft_answer", "") or ""
    clean, audit = strip_links(draft, cfg)

    n = audit["stripped_count"]
    note = f"{n} url(s) stripped" if n else "clean"

    return {
        "answer": clean,
        "link_audit": audit,
        "messages": [AIMessage(content=clean)],
        "timings": {"sanitize": round(time.perf_counter() - started, 4)},
        "trace": [f"sanitize: {note}"],
    }


__all__ = ["sanitize"]
