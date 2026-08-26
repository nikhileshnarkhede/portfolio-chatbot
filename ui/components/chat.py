"""
chat.py - message rendering and the streaming turn.

Two things here are worth knowing.

**The UI keeps its own message list.** `st.session_state.transcript` is what the
user sees; the graph's checkpointer holds the model's working history. They
diverge on purpose: summarization trims the graph's `messages` down to the last
few turns, and the person reading the page should not watch their conversation
disappear because the token budget needed trimming.

**Streaming goes through the graph, not around it.** The callback is passed via
`config["configurable"]["on_chunk"]`, so tokens render live while the exact same
node code path runs that an eval run uses. The alternative - calling the LLM
directly from the UI for streaming - is how the original ended up with the
pipeline duplicated inside the render loop.
"""

from __future__ import annotations

import time

import streamlit as st

from portfolio_chatbot.config import AppConfig
from portfolio_chatbot.graph import run_turn

CURSOR = "▌"


def init_state() -> None:
    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("last_state", None)
    st.session_state.setdefault("turn_count", 0)


def reset(thread_id: str) -> None:
    st.session_state.transcript = []
    st.session_state.last_state = None
    st.session_state.pending_question = None
    st.session_state.turn_count = 0
    st.session_state.thread_id = thread_id


def quick_actions(cfg: AppConfig) -> None:
    """Starter buttons, shown only on an empty conversation."""
    if st.session_state.transcript:
        return
    actions = cfg.ui.quick_actions
    for col, action in zip(st.columns(len(actions)), actions):
        with col:
            if st.button(action.label, use_container_width=True, help=action.question):
                st.session_state.pending_question = action.question


def render_history(cfg: AppConfig) -> None:
    for message in st.session_state.transcript:
        avatar = cfg.ui.assistant_avatar if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def take_question(cfg: AppConfig) -> str | None:
    """Whatever the user asked this rerun: typed input or a quick-action click."""
    typed = st.chat_input("Ask something about my background...")
    if typed:
        return typed
    pending = st.session_state.pending_question
    st.session_state.pending_question = None
    return pending


def answer(cfg: AppConfig, question: str, thread_id: str) -> None:
    """Run one turn, streaming into the page."""
    st.session_state.transcript.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=cfg.ui.assistant_avatar):
        status = st.empty()
        placeholder = st.empty()
        buffer: list[str] = []
        delay = cfg.ui.stream_delay

        status.markdown("_Thinking..._")

        def on_chunk(piece: str) -> None:
            # Clear the status on the FIRST token, not when the turn finishes.
            # `st.spinner` wraps a whole block, so it used to sit there spinning
            # while the answer was already streaming underneath it - the reply
            # was visibly arriving and the page still said "Thinking". Perceived
            # latency is time-to-first-token, and this makes the UI report that
            # number instead of the total.
            if not buffer:
                status.empty()
            buffer.append(piece)
            placeholder.markdown("".join(buffer) + CURSOR)
            if delay:
                time.sleep(delay)

        final = run_turn(cfg, question, thread_id=thread_id, on_chunk=on_chunk)
        status.empty()   # non-streaming mode never fires on_chunk

        # Render the SANITIZED answer, not the streamed buffer. The buffer is
        # raw model output; the guard runs after generation, so a forged link
        # can be visible mid-stream and must not survive the final paint.
        placeholder.markdown(final["answer"])

        if final.get("error"):
            _failure_notice(final)

    st.session_state.transcript.append({"role": "assistant", "content": final["answer"]})
    st.session_state.last_state = final
    st.session_state.turn_count += 1

    if cfg.ui.show_debug:
        _field_rating(cfg, question, final)


def _failure_notice(final) -> None:
    """Say which failure this was, rather than assuming the common one.

    This used to hardcode "every model is at capacity" for ANY error, which
    contradicted the reply the graph had already written directly above it -
    `generate` distinguishes rate limits from everything else and only says
    "rate limits" when `is_rate_limited` agreed. So the page could show a
    generic failure message and a rate-limit explanation at the same time, and
    the explanation was the wrong one.

    Guessing costs real time here. "At capacity" means wait; a rejected key or
    a decommissioned model means the wait will never end.
    """
    from portfolio_chatbot.llm.fallback import is_rate_limited

    error = str(final.get("error") or "")
    attempts = final.get("model_attempts") or []
    lowered = error.lower()

    if is_rate_limited(error):
        st.warning(
            f"Rate limited on every model tried ({len(attempts)}). "
            "This one does clear on its own - try again in a few minutes."
        )
    elif "401" in error or "invalid api key" in lowered or "authentication" in lowered:
        st.error(
            "Groq rejected the API key. It is reaching the app, so this is the "
            "key itself rather than a missing one — check for surrounding "
            "quotes or a stray newline if it came from a `.env` file."
        )
    elif "404" in error or "does not exist" in lowered or "decommission" in lowered:
        st.error(
            "Groq does not recognise one of the models in `llm.model_chain`. "
            "Model IDs get retired; check the chain against "
            "https://console.groq.com/docs/models."
        )
    else:
        st.error("The answer could not be generated. The details are below.")

    if error:
        with st.expander("What went wrong"):
            st.code(error, language="text")
            if attempts:
                st.caption("Models tried, in order: "
                           + ", ".join(a.get("model", "?") for a in attempts))


def _field_rating(cfg: AppConfig, question: str, final) -> None:
    """Inline 0-10 rating, debug mode only.

    Logged separately from the reference-set ratings and never counted toward a
    gate: you can see which config produced this answer, so it is not blind. Its
    value is as a source of NEW reference questions - a field rating of 3 is a
    case the golden set is missing.
    """
    from eval import ratings as ratings_mod

    with st.expander("Rate this answer", expanded=False):
        score = st.slider("0-10", 0, 10, 7, key=f"fr_{st.session_state.turn_count}")
        st.caption(ratings_mod.rubric_for(score))
        note = st.text_input("Note", key=f"frn_{st.session_state.turn_count}")
        if st.button("Save rating", key=f"frb_{st.session_state.turn_count}"):
            ratings_mod.save_field_rating(
                question, final.get("answer", ""), score, cfg.run_fingerprint, note)
            st.success("Logged. Field ratings inform the reference set; "
                       "they never gate a release.")


__all__ = ["init_state", "reset", "quick_actions", "render_history", "take_question", "answer"]
