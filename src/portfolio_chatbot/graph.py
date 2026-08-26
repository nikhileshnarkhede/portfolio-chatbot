"""
graph.py - topology only.

This file wires nodes together and compiles. It contains no retrieval logic, no
prompt handling, and no LLM calls; if you find yourself adding an `if` about
*what* a step does rather than *whether* it runs, it belongs in a node.

    expand_query -> route -> retrieve -> generate -> sanitize -> [summarize] -> END

Two orderings are load-bearing:

* **expand_query before route.** Routing normally reads the raw question, so
  the order looks arbitrary - until you set `retrieval.routing.route_on` to
  "expanded_question", which is a supported experiment. Putting expansion first
  makes both settings work without changing the graph.

* **sanitize before summarize.** Summarization condenses `messages`, and the
  reply only enters `messages` after the guard has stripped forged URLs.
  Reversed, a hallucinated link would be summarized into the rolling history
  and outlive the turn that produced it.

The edge into `summarize` is conditional. The node no-ops below the threshold
anyway, but skipping it outright keeps short conversations - which is most eval
cases - off a code path that can make an LLM call.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .config import AppConfig
from .memory.checkpointer import build_checkpointer
from .nodes import CONFIG_KEY
from .nodes.expand_query import expand_query
from .nodes.generate import generate
from .nodes.retrieve import retrieve
from .nodes.route import route
from .nodes.sanitize import sanitize
from .nodes.summarize import summarize
from .state import GraphState, new_turn

#: Compiled graphs, keyed by run fingerprint. Compilation is cheap but a
#: Streamlit rerun happens on every keystroke, and rebuilding the topology per
#: rerun is pure waste.
_GRAPHS: dict[str, Any] = {}


def build_graph(cfg: AppConfig, *, checkpointer: Any | None = "auto"):
    """Compile the pipeline for `cfg`.

    Pass `checkpointer=None` for a stateless graph, or an instance to supply
    your own. The default builds whatever `memory.checkpointer` names.
    """
    builder = StateGraph(GraphState)

    builder.add_node("expand_query", expand_query)
    builder.add_node("route", route)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("sanitize", sanitize)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "expand_query")
    builder.add_edge("expand_query", "route")
    builder.add_edge("route", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "sanitize")

    threshold = cfg.memory.summarize_after_n_messages
    enabled = cfg.memory.enabled

    def after_sanitize(state: GraphState) -> str:
        if not enabled:
            return END
        return "summarize" if len(state.get("messages") or []) > threshold else END

    builder.add_conditional_edges("sanitize", after_sanitize,
                                  {"summarize": "summarize", END: END})
    builder.add_edge("summarize", END)

    saver = build_checkpointer(cfg) if checkpointer == "auto" else checkpointer
    return builder.compile(checkpointer=saver)


def get_graph(cfg: AppConfig):
    """Compiled graph for this config, cached by run fingerprint."""
    key = cfg.run_fingerprint
    if key not in _GRAPHS:
        _GRAPHS[key] = build_graph(cfg)
    return _GRAPHS[key]


def runnable_config(cfg: AppConfig, thread_id: str = "default",
                    on_chunk: Callable[[str], None] | None = None) -> dict:
    """The RunnableConfig every node expects."""
    configurable: dict[str, Any] = {CONFIG_KEY: cfg, "thread_id": thread_id}
    if on_chunk is not None:
        configurable["on_chunk"] = on_chunk
    return {"configurable": configurable}


def run_turn(cfg: AppConfig, question: str, *, thread_id: str = "default",
             app: Any | None = None, on_chunk: Callable[[str], None] | None = None,
             summary: str | None = None) -> GraphState:
    """Run one question end to end and return the final state.

    The human message is added here rather than inside a node: the graph's job
    starts once the question is already part of the conversation.
    """
    from langchain_core.messages import HumanMessage

    app = app or get_graph(cfg)
    state = new_turn(
        question,
        experiment_name=cfg.run.experiment_name,
        run_fingerprint=cfg.run_fingerprint,
        index_fingerprint=cfg.index_fingerprint,
        summary=summary,
    )
    state["messages"] = [HumanMessage(content=question)]
    return app.invoke(state, runnable_config(cfg, thread_id, on_chunk))


def clear_cache() -> None:
    _GRAPHS.clear()


def render_mermaid(cfg: AppConfig) -> str:
    """Mermaid source for the compiled topology, for docs and the debug panel."""
    return get_graph(cfg).get_graph().draw_mermaid()


__all__ = [
    "build_graph", "get_graph", "run_turn", "runnable_config",
    "render_mermaid", "clear_cache",
]
