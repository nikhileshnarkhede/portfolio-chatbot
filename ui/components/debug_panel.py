"""
debug_panel.py - inspect the last turn.

Everything shown here is read straight out of the final `GraphState`. That is
the payoff of the rule in `state.py` that nothing may be computed and thrown
away: the panel needs no instrumentation of its own, and what you see here is
exactly what `runs/` will contain.

Hidden unless `ui.show_debug` is true, so it never reaches a recruiter.
"""

from __future__ import annotations

import streamlit as st

from portfolio_chatbot.config import AppConfig
from portfolio_chatbot.state import GraphState, chunk_refs


def render(cfg: AppConfig, state: GraphState | None) -> None:
    if not state:
        st.caption("Ask something to populate the debug panel.")
        return

    _pipeline(state)
    _retrieval(cfg, state)
    _generation(state)
    _safety(state)
    _memory(state)
    _identity(state)


def _pipeline(state: GraphState) -> None:
    timings = state.get("timings") or {}
    total = sum(timings.values())
    with st.expander(f"Pipeline — {total:.2f}s", expanded=True):
        for line in state.get("trace") or []:
            st.text(line)
        if timings:
            st.bar_chart(timings, horizontal=True, height=200)


def _retrieval(cfg: AppConfig, state: GraphState) -> None:
    route = state.get("route") or {}
    mode = state.get("retrieval_mode", "?")
    label = route.get("rule_name") or "no match"

    with st.expander(f"Retrieval — {mode}, route: {label}"):
        cols = st.columns(3)
        cols[0].metric("Chunks", len(state.get("documents") or []))
        cols[1].metric("Context chars", state.get("context_chars", 0))
        cols[2].metric("Truncated", "yes" if state.get("context_truncated") else "no")

        if state.get("context_truncated"):
            st.warning(
                f"Context hit max_context_chars={cfg.retrieval.context.max_context_chars}. "
                f"Chunks the retriever ranked as relevant were cut off."
            )

        if route.get("matched"):
            st.caption(f"Matched '{route.get('matched_keyword')}' → {route.get('chunk_types')}")

        refs = chunk_refs(state)
        if refs:
            st.dataframe(refs, use_container_width=True, hide_index=True)

        st.text_area("Context sent to the model", state.get("context", ""),
                     height=220, disabled=True)


def _generation(state: GraphState) -> None:
    attempts = state.get("model_attempts") or []
    used = state.get("model_used") or "none"
    header = f"Generation — {used}"
    if len(attempts) > 1:
        header += f" (fell back past {len(attempts) - 1})"

    with st.expander(header):
        if attempts:
            st.dataframe(attempts, use_container_width=True, hide_index=True)
        if state.get("error"):
            st.error(state["error"])
        st.text_area("Raw model output (pre-sanitize)", state.get("draft_answer", ""),
                     height=160, disabled=True)


def _safety(state: GraphState) -> None:
    audit = state.get("link_audit") or {}
    stripped = audit.get("stripped") or []
    header = f"URL guard — {len(stripped)} stripped"

    with st.expander(header, expanded=bool(stripped)):
        if stripped:
            st.error(
                "The model produced links that are not in the resume. The guard "
                "removed them, but this is a prompt failure worth recording."
            )
            for url in stripped:
                st.code(url, language=None)
        else:
            st.success("No forged links this turn.")
        if audit.get("kept"):
            st.caption("Kept: " + ", ".join(audit["kept"]))


def _memory(state: GraphState) -> None:
    summary = state.get("summary") or ""
    fired = state.get("summarized_this_turn")
    with st.expander(f"Memory — {'summarized this turn' if fired else 'unchanged'}"):
        st.caption(f"Messages in graph history: {len(state.get('messages') or [])}")
        st.text_area("Rolling summary", summary or "(none yet)", height=110, disabled=True)


def _identity(state: GraphState) -> None:
    with st.expander("Run identity"):
        st.json({
            "turn_id": state.get("turn_id"),
            "experiment": state.get("experiment_name"),
            "run_fingerprint": state.get("run_fingerprint"),
            "index_fingerprint": state.get("index_fingerprint"),
        })


__all__ = ["render"]
