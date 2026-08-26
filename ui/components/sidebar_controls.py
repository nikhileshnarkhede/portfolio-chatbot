"""
sidebar_controls.py - switch experiments and inspect the running config.

This is the piece that turns the app into an evaluation instrument rather than
just a demo. Pick an experiment from the sidebar and the next question runs
through that config's chunking, retrieval, prompt and model settings - no
restart, no code edit.

Two rules it enforces, both of which would otherwise produce quietly wrong
results:

* **Switching experiments starts a new thread.** Conversation history built
  under one prompt is not a clean input to another; carrying it over would
  contaminate the comparison you just set up.
* **A config whose index has not been built is reported, not silently
  answered.** Falling back to another experiment's vectors is exactly the kind
  of error that produces plausible numbers from the wrong setup.
"""

from __future__ import annotations

import streamlit as st

from portfolio_chatbot.config import AppConfig, list_experiments, load_config

DEFAULT_LABEL = "default (configs/default.json)"


def render(cfg: AppConfig) -> tuple[AppConfig, bool]:
    """Draw the sidebar. Returns (config_to_use, experiment_changed)."""
    changed = False

    with st.sidebar:
        st.subheader("Experiment")

        options = [DEFAULT_LABEL] + list_experiments()
        current = st.session_state.get("experiment") or DEFAULT_LABEL
        index = options.index(current) if current in options else 0

        chosen = st.selectbox(
            "Config", options, index=index,
            help="Each experiment is a delta over configs/default.json.",
        )
        if chosen != current:
            st.session_state.experiment = chosen
            changed = True

        cfg = load_config(None if chosen == DEFAULT_LABEL else chosen)

        _index_status(cfg)
        _summary(cfg)

        st.divider()
        st.session_state.show_debug = st.toggle(
            "Debug panel", value=st.session_state.get("show_debug", cfg.ui.show_debug),
            help="Route, retrieved chunks, raw model output, URL guard, timings.",
        )
        if st.button("Reset conversation", use_container_width=True):
            changed = True

        _topology(cfg)

    return cfg, changed


def _index_status(cfg: AppConfig) -> None:
    if cfg.index_path.exists():
        st.success(f"Index ready · `{cfg.index_fingerprint}`")
        return
    st.error(f"No index for `{cfg.index_fingerprint}`")
    name = cfg.run.experiment_name
    st.code(
        "python scripts/ingest.py"
        + (f" --experiment {name}" if name != "default" else ""),
        language="bash",
    )


def _summary(cfg: AppConfig) -> None:
    with st.expander("Active settings"):
        st.markdown(f"""
- **chunking** `{cfg.ingestion.strategy}` @ {cfg.ingestion.split.max_chunk_chars} chars
- **retrieval** k={cfg.retrieval.filtered.k} / mmr k={cfg.retrieval.mmr.k}
- **expansion** `{cfg.retrieval.query_expansion.mode}`
- **prompt** `{cfg.prompts.system}`
- **model** `{cfg.llm.primary_model}` (temp {cfg.llm.temperature})
- **memory** summarize past {cfg.memory.summarize_after_n_messages}, keep {cfg.memory.keep_last_n}
""")
        st.caption(f"run `{cfg.run_fingerprint}` · index `{cfg.index_fingerprint}`")


def _topology(cfg: AppConfig) -> None:
    with st.expander("Graph"):
        try:
            from portfolio_chatbot.graph import render_mermaid
            st.code(render_mermaid(cfg), language="mermaid")
        except Exception as exc:  # pragma: no cover - diagram is a nicety
            st.caption(f"Diagram unavailable: {exc}")


__all__ = ["render", "DEFAULT_LABEL"]
