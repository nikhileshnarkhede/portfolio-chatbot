"""
Tests for the compiled graph.

These are the integration tests: real topology, real state reducers, real
checkpointer, a real FAISS index over the real resume - with only the embedder
and the chat model faked. Everything the node tests cover in isolation has to
still hold once LangGraph is merging partial updates between steps, and a few
properties only exist at this level:

* the sanitized answer, not the draft, is what enters conversation history;
* per-turn diagnostics stay per-turn across a long threaded conversation;
* summarization fires at the threshold and its summary survives to later turns.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from portfolio_chatbot import graph as G
from portfolio_chatbot.config import load_config

ALLOWED = "https://github.com/nikhileshnarkhede"
FORGED = "https://evil.example.com"


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, _prompt):
        return AIMessage(content="Earlier turns covered his projects.")

    def stream(self, _prompt):
        for word in self.reply.split(" "):
            yield AIMessage(content=word + " ")


@pytest.fixture
def wired(cfg, fake_index, monkeypatch):
    """A graph with a real index and a fake model. Returns a runner."""
    reply = f"I built things, see [GitHub]({ALLOWED}) and [x]({FORGED})."
    for module in ("generate", "summarize"):
        monkeypatch.setattr(f"portfolio_chatbot.nodes.{module}.build_llm",
                            lambda c, model=None: FakeLLM(reply))
    G.clear_cache()
    yield lambda q, **kw: G.run_turn(cfg, q, **kw)
    G.clear_cache()


# ---------------------------------------------------------------- topology

def test_graph_compiles(cfg):
    assert G.build_graph(cfg, checkpointer=None) is not None


def test_every_node_is_present(cfg):
    nodes = set(G.build_graph(cfg, checkpointer=None).get_graph().nodes)
    assert {"expand_query", "route", "retrieve", "generate", "sanitize", "summarize"} <= nodes


def test_graphs_are_cached_per_run_fingerprint(cfg):
    G.clear_cache()
    assert G.get_graph(cfg) is G.get_graph(cfg)
    assert G.get_graph(load_config("exp003_prompt_v2")) is not G.get_graph(cfg)


def test_mermaid_renders(cfg):
    assert "expand_query" in G.render_mermaid(cfg)


# ---------------------------------------------------------------- one full turn

def test_full_turn_populates_the_pipeline(wired):
    out = wired("Tell me about your projects", thread_id="t1")
    assert out["route"]["rule_name"] == "projects"
    assert out["retrieval_mode"] == "filtered"
    assert out["documents"]
    assert out["answer"]
    assert len(out["trace"]) == 5


def test_every_node_records_a_timing(wired):
    timings = wired("Tell me about your projects", thread_id="t1")["timings"]
    assert set(timings) == {"expand_query", "route", "retrieve", "generate", "sanitize"}


def test_streaming_callback_receives_chunks(wired):
    seen = []
    wired("Tell me about your projects", thread_id="t1", on_chunk=seen.append)
    assert len(seen) > 1


def test_draft_and_answer_both_survive_the_graph(wired):
    out = wired("Tell me about your projects", thread_id="t1")
    assert FORGED in out["draft_answer"]
    assert FORGED not in out["answer"]
    assert ALLOWED in out["answer"]
    assert out["link_audit"]["stripped"] == [FORGED]


def test_history_stores_the_sanitized_answer(wired):
    """A forged link in history would become an established fact of the chat."""
    out = wired("Tell me about your projects", thread_id="t1")
    assert FORGED not in out["messages"][-1].content
    assert out["messages"][-1].content == out["answer"]


def test_expansion_runs_before_routing(wired):
    """Order matters when routing.route_on is switched to the expanded text."""
    trace = wired("what are your skills", thread_id="t1")["trace"]
    assert trace[0].startswith("expand")
    assert trace[1].startswith("route")


# ---------------------------------------------------------------- threading

def test_history_accumulates_across_turns(wired):
    for i in range(3):
        out = wired(f"question {i}", thread_id="shared")
    assert len(out["messages"]) == 6


def test_threads_are_isolated(wired):
    wired("first", thread_id="a")
    wired("first", thread_id="a")
    other = wired("first", thread_id="b")
    assert len(other["messages"]) == 2


@pytest.mark.parametrize("turns", [2, 4, 6])
def test_per_turn_diagnostics_do_not_leak(wired, turns):
    """REGRESSION: accumulating fields must describe one turn, not the thread."""
    for i in range(turns):
        out = wired(f"question {i}", thread_id="leak")
    assert len(out["model_attempts"]) == 1
    assert len(out["trace"]) == 5


# ---------------------------------------------------------------- summarization

def test_summarization_fires_at_the_threshold_and_persists(wired, cfg):
    fired_on = None
    for i in range(1, 9):
        out = wired(f"question {i}", thread_id="long")
        if out.get("summarized_this_turn"):
            fired_on = fired_on or i

    assert fired_on == 7, "14 messages exceeds the threshold of 12 on turn 7"
    assert out["summary"], "the summary must survive into later turns"


def test_summarization_trims_to_keep_last_n(wired, cfg):
    for i in range(7):
        out = wired(f"question {i}", thread_id="trim")
    assert len(out["messages"]) == cfg.memory.keep_last_n


def test_summarize_is_skipped_when_memory_is_disabled(cfg, fake_index, monkeypatch):
    off = load_config(overrides=["memory.enabled=false"])
    monkeypatch.setattr("portfolio_chatbot.nodes.generate.build_llm",
                        lambda c, model=None: FakeLLM("hello"))
    G.clear_cache()
    for i in range(8):
        out = G.run_turn(off, f"q{i}", thread_id="nomem")
    assert not out.get("summarized_this_turn")
    assert len(out["messages"]) == 16
    G.clear_cache()


def test_new_turn_does_not_wipe_the_persisted_summary(wired):
    """REGRESSION: a default summary="" would erase summarization every turn."""
    for i in range(8):
        out = wired(f"question {i}", thread_id="keep")
    assert out["summary"]
