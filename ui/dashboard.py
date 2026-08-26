"""
dashboard.py - the tuning and evaluation console.

    streamlit run ui/dashboard.py

Six tabs, in the order you actually work: tune the parameters, run the suites,
read the scorecard, compare against the baseline, rate the answers a metric
cannot judge, and apply what survives.

Two rules shape the whole thing.

**Colour encodes gate state, not magnitude.** A metric at 0.82 against a 0.90
threshold is red; the same 0.82 against a 0.80 threshold is green. The number
alone never tells you whether to act.

**A delta smaller than the minimum detectable effect renders as
*inconclusive*, never as an improvement.** That is the whole reason the noise
floor exists, and a dashboard that shows a green arrow for movement inside its
own measurement error is worse than one showing nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.runner import DEFAULT_DATASETS  # noqa: E402
from portfolio_chatbot.config import DEFAULT_CONFIG_PATH, AppConfig  # noqa: E402
from ui.dash import panels, state  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Chatbot Eval Console", page_icon="⚖",
                       layout="wide", initial_sidebar_state="expanded")
    panels.inject_styles()
    state.init()

    base = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    working = st.session_state.working_config

    try:
        cfg: AppConfig | None = load_config_from(working)
        config_error = None
    except Exception as exc:
        cfg, config_error = None, str(exc)

    panels.sidebar(cfg, working, base, config_error)

    st.title("Evaluation console")
    if config_error:
        st.error(f"This configuration is invalid, so nothing below can run.\n\n{config_error}")

    tabs = st.tabs(["1 · Tune", "2 · Run", "3 · Scorecard", "4 · Compare",
                    "5 · Rate", "6 · Apply", "⚙ Thresholds"])

    with tabs[0]:
        panels.tune(working, base, cfg)
    with tabs[1]:
        panels.run_tab(cfg, DEFAULT_DATASETS)
    with tabs[2]:
        panels.scorecard(cfg)
    with tabs[3]:
        panels.compare(cfg)
    with tabs[4]:
        panels.rate(cfg)
    with tabs[5]:
        panels.apply_tab(cfg, working, base)
    with tabs[6]:
        panels.thresholds()


def load_config_from(working: dict) -> AppConfig:
    """Validate the in-progress config without writing it anywhere.

    The dashboard edits a plain dict and validates on every rerun, so an invalid
    combination (fetch_k below k, keep_last_n above the summarize trigger) is
    reported the moment you create it rather than when you try to run.
    """
    from portfolio_chatbot.config import AppConfig as _AppConfig
    from portfolio_chatbot.config import _strip_meta
    return _AppConfig.model_validate(_strip_meta(working))


if __name__ == "__main__":
    main()
