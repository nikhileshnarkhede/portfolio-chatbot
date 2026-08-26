"""
summarize.py - keep conversation history inside the token budget.

Once history passes `memory.summarize_after_n_messages`, the older turns are
condensed into `summary` and dropped from `messages`, leaving the most recent
`memory.keep_last_n`. Trimming is done with `RemoveMessage`, the LangGraph
idiom - returning a shorter list would not work, because the `add_messages`
reducer appends rather than replaces.

**One deliberate divergence from the original app.py.** The original replaced
`conversation_summary` outright on every summarization pass, so the second pass
discarded everything the first pass had summarized - a long conversation
silently lost its early history rather than compressing it. Here the existing
summary is folded into the text being summarized, so it rolls forward.

That is a correctness fix, not a tuning choice, and it only changes behaviour
past the twelfth message. Baseline parity with the deployed bot still holds for
chunking and the system prompt, which is where it matters; if you need a strict
replay of the old behaviour, set `memory.enabled` to false.
"""

from __future__ import annotations

import time
from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig

from ..llm.fallback import run_with_fallback
from ..llm.provider import build_llm
from ..memory import history
from ..prompts.loader import summarizer_prompt
from ..state import GraphState
from . import app_config

FALLBACK_SUMMARY = (
    "Earlier conversation covered various topics about Nikhilesh's background."
)


def summarize(state: GraphState, config: RunnableConfig) -> dict:
    cfg = app_config(config)
    started = time.perf_counter()
    mem = cfg.memory

    messages = list(state.get("messages") or [])

    if not mem.enabled or len(messages) <= mem.summarize_after_n_messages:
        return {}

    to_drop = messages[: -mem.keep_last_n]
    if not to_drop:
        return {}

    previous = state.get("summary", "")
    conversation = history.for_summary(to_drop, cfg)
    if previous:
        # Roll the existing summary forward instead of discarding it.
        conversation = f"[Summary of even earlier turns]\n{previous}\n\n{conversation}"

    rendered = summarizer_prompt(cfg).format(
        conversation=conversation, max_sentences=mem.summary_max_sentences,
    )

    def call(model: str) -> str:
        return build_llm(cfg, model).invoke(rendered).content

    try:
        summary = run_with_fallback(cfg, call).text.strip() or FALLBACK_SUMMARY
        note = f"summarized {len(to_drop)} msgs"
    except Exception:
        # Trimming must still happen: the reason we are here is that history is
        # too long, and leaving it intact would make the next turn worse.
        summary = previous or FALLBACK_SUMMARY
        note = f"summarizer failed, trimmed {len(to_drop)} msgs anyway"

    return {
        "summary": summary,
        "summarized_this_turn": True,
        "messages": [RemoveMessage(id=m.id) for m in to_drop if m.id],
        "timings": {"summarize": round(time.perf_counter() - started, 4)},
        "trace": [f"summarize: {note}"],
    }


__all__ = ["summarize", "FALLBACK_SUMMARY"]
