"""
expand_query.py - widen the text used for ranking.

Three modes, from `retrieval.query_expansion.mode`:

``keyword_rules``  the original app.py behaviour: append domain terms when the
                   question matches a rule. Free, deterministic, no LLM call.
``llm``            rewrite the question with the model, using the
                   `query_expansion` prompt. Can resolve follow-up pronouns
                   ("tell me more about that one") into standalone queries,
                   which keyword rules fundamentally cannot. Costs a call and
                   adds a second place the model can go wrong.
``none``           passthrough. The control arm - worth running, because
                   expansion is widely assumed to help and that assumption is
                   exactly the sort of thing this project exists to check.

The expanded text is used for RANKING only. Routing reads the raw question.
"""

from __future__ import annotations

import time
from langchain_core.runnables import RunnableConfig
from ..config import AppConfig
from ..state import GraphState
from . import app_config


def _keyword_expand(question: str, cfg: AppConfig) -> tuple[str, str | None]:
    lowered = question.lower()
    for rule in cfg.retrieval.query_expansion.rules:
        if any(k in lowered for k in rule.keywords):
            return f"{question} {rule.append}", rule.append
    return question, None


def _llm_expand(question: str, cfg: AppConfig) -> str:
    from ..llm.fallback import run_with_fallback
    from ..llm.provider import build_llm
    from ..prompts.loader import query_expansion_prompt

    rendered = query_expansion_prompt(cfg).format(question=question)

    def call(model: str) -> str:
        return build_llm(cfg, model).invoke(rendered).content

    # A failed rewrite must not fail the turn - fall back to the raw question.
    try:
        return run_with_fallback(cfg, call).text.strip() or question
    except Exception:
        return question


def expand_query(state: GraphState, config: RunnableConfig) -> dict:
    cfg = app_config(config)
    started = time.perf_counter()
    question = state.get("question", "")
    qe = cfg.retrieval.query_expansion

    if not qe.enabled or qe.mode == "none":
        expanded, note = question, "disabled"
    elif qe.mode == "keyword_rules":
        expanded, appended = _keyword_expand(question, cfg)
        note = f"+'{appended}'" if appended else "no rule matched"
    elif qe.mode == "llm":
        expanded = _llm_expand(question, cfg)
        note = "llm rewrite" if expanded != question else "llm rewrite (unchanged)"
    else:  # pragma: no cover - config validation catches this
        raise ValueError(f"Unknown query expansion mode: {qe.mode!r}")

    return {
        "expanded_question": expanded,
        "timings": {"expand_query": round(time.perf_counter() - started, 4)},
        "trace": [f"expand: {note}"],
    }


__all__ = ["expand_query"]
