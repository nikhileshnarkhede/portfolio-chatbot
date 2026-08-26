"""
generate.py - render the prompt, call the model, keep the raw output.

What this node writes is `draft_answer`, never `answer`. The distinction is
load-bearing: `sanitize` runs next and produces `answer`, and keeping both is
the only way to measure how often the model attempts a URL the allowlist has
to strip. A node that wrote straight to `answer` would destroy that evidence.

Failure is handled here rather than allowed to escape. If every model in the
chain fails, the turn still produces a usable reply and an `error` field - an
eval run over a hundred questions must not abort on question 43 because the
free tier ran dry, and the run log needs to record the failure as data.
"""

from __future__ import annotations

import time
from typing import Callable

from langchain_core.runnables import RunnableConfig

from ..llm.fallback import AllModelsFailed, run_with_fallback
from ..llm.provider import build_llm
from ..memory import history
from ..prompts.loader import system_prompt
from ..state import GraphState
from . import app_config

#: Shown to the user when the whole chain is exhausted. Deliberately in
#: Nikhilesh's voice - it is still his chatbot speaking.
RATE_LIMITED_REPLY = (
    "I've hit rate limits across all my backup models right now. "
    "Please try again in a few minutes, or contact me directly at "
    "narkhede.nikhilesh@gmail.com."
)
GENERIC_FAILURE_REPLY = (
    "I ran into a problem answering that one. Please try again, or reach me "
    "directly at narkhede.nikhilesh@gmail.com."
)


def generate(state: GraphState, config: RunnableConfig, *,
             on_chunk: Callable[[str], None] | None = None) -> dict:
    """Produce `draft_answer`.

    `on_chunk` renders tokens as they arrive. Inside a compiled graph a node is
    called as `node(state, config)`, so the UI passes its callback through
    `config["configurable"]["on_chunk"]` instead; the keyword argument is for
    calling this node directly. An eval run passes neither and takes exactly
    the same code path with no display attached.
    """
    cfg = app_config(config)
    started = time.perf_counter()

    if on_chunk is None and isinstance(config, dict):
        on_chunk = config.get("configurable", {}).get("on_chunk")

    rendered = system_prompt(cfg).format(
        context=state.get("context", ""),
        question=state.get("question", ""),
        chat_history=history.for_prompt(
            state.get("messages") or [], state.get("summary", ""), cfg
        ),
    )

    def call(model: str):
        # Hand the raw chunks/message to the fallback rather than mapping to
        # .content here. Token usage rides on the message object, so extracting
        # text early would throw it away before it could be recorded - and the
        # loss would be silent, showing up only as usage_reported = 0.
        llm = build_llm(cfg, model)
        return llm.stream(rendered) if cfg.llm.streaming else llm.invoke(rendered)

    start_index = int(state.get("model_start_index", 0) or 0)

    try:
        result = run_with_fallback(cfg, call, start_index=start_index, on_chunk=on_chunk)
    except AllModelsFailed as exc:
        reply = RATE_LIMITED_REPLY if exc.rate_limited else GENERIC_FAILURE_REPLY
        return {
            "draft_answer": reply,
            "model_used": "",
            "model_attempts": exc.attempts,
            "error": str(exc)[:500],
            "timings": {"generate": round(time.perf_counter() - started, 4)},
            "trace": [f"generate: FAILED after {len(exc.attempts)} model(s)"],
        }

    usage = result.usage(cfg)
    note = f"{result.model_used}"
    if result.fell_back:
        note += f" (fell back past {len(result.attempts) - 1})"
    if usage["available"]:
        note += f" [{usage['input_tokens']}+{usage['output_tokens']} tok]"

    return {
        "draft_answer": result.text,
        "token_usage": usage,
        "ttft_s": result.ttft_s,
        "model_used": result.model_used,
        "model_attempts": result.attempts,
        "model_start_index": result.next_start_index,
        "error": None,
        "timings": {"generate": round(time.perf_counter() - started, 4)},
        "trace": [f"generate: {note}"],
    }


__all__ = ["generate", "RATE_LIMITED_REPLY", "GENERIC_FAILURE_REPLY"]
