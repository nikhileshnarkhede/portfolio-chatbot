"""
nodes/ - one LangGraph node per file.

Every node has the same shape:

    def node(state: GraphState, config: RunnableConfig) -> dict

It reads what it needs from the state, gets its parameters from the config via
`app_config(config)`, and returns a **partial** dict. LangGraph merges that
using the reducers declared in `state.py`. A node never mutates the state it is
handed, and never returns the whole state.

Nodes are deliberately thin. The real work lives in `tools/`, `ingestion/` and
`llm/`, so it can be tested without constructing a graph at all.
"""

from __future__ import annotations

from typing import Any

from ..config import AppConfig

CONFIG_KEY = "app_config"


def app_config(config: Any) -> AppConfig:
    """Pull the AppConfig out of LangGraph's RunnableConfig.

    The config travels here rather than in the state because state is
    serialized into every checkpoint; see the note in `state.py`.
    """
    cfg = None
    if isinstance(config, dict):
        cfg = config.get("configurable", {}).get(CONFIG_KEY)
    if cfg is None:
        raise KeyError(
            f"No AppConfig on the RunnableConfig. Invoke the graph with "
            f'config={{"configurable": {{"{CONFIG_KEY}": cfg, "thread_id": ...}}}}.'
        )
    return cfg


__all__ = ["app_config", "CONFIG_KEY"]
