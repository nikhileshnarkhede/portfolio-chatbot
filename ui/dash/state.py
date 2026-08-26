"""
state.py - dashboard session state.

The working config is a plain dict rather than an AppConfig, because the user
is allowed to hold an *invalid* configuration mid-edit: dragging `fetch_k`
below `k` on the way to a valid pair must not raise. Validation happens on
every rerun and is reported, not enforced.
"""

from __future__ import annotations

import copy
import json

import streamlit as st

from portfolio_chatbot.config import DEFAULT_CONFIG_PATH


def init() -> None:
    if "working_config" not in st.session_state:
        reset_to_default()
    st.session_state.setdefault("last_report", None)
    st.session_state.setdefault("last_gate", None)
    st.session_state.setdefault("last_noise", None)
    st.session_state.setdefault("rate_index", 0)


def reset_to_default() -> None:
    st.session_state.working_config = json.loads(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def working() -> dict:
    return st.session_state.working_config


def update(new_config: dict) -> None:
    st.session_state.working_config = copy.deepcopy(new_config)


def clear_results() -> None:
    """Results belong to a config. Changing the config invalidates them."""
    st.session_state.last_report = None
    st.session_state.last_gate = None


__all__ = ["init", "reset_to_default", "working", "update", "clear_results"]
