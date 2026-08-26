"""
Tests for the graph state contract.

The multi-turn tests here are not academic. During development, `new_turn`
cleared its accumulating fields by assigning `[]` and `{}`. That silently did
nothing: an appending reducer is called as `reducer(existing, update)`, so an
empty update appends nothing and leaves the previous turn's entries in place.
With a checkpointer, turn 3 reported three `model_attempts` and three `trace`
lines, so any per-turn metric computed from a run log would have described the
whole conversation instead of the turn. `test_accumulators_do_not_leak_across_turns`
is the regression guard for exactly that.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from portfolio_chatbot.state import (
    RESET,
    EVAL_FIELDS,
    GraphState,
    append_or_reset,
    chunk_refs,
    merge_or_reset,
    new_turn,
)


# ---------------------------------------------------------------- reducers

def test_append_or_reset_appends():
    assert append_or_reset([1], [2, 3]) == [1, 2, 3]


def test_append_or_reset_handles_empty_left():
    assert append_or_reset(None, [1]) == [1]


def test_append_or_reset_clears_on_reset():
    assert append_or_reset([1, 2], RESET) == []


def test_merge_or_reset_merges_disjoint_keys():
    assert merge_or_reset({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_merge_or_reset_right_wins():
    assert merge_or_reset({"a": 1}, {"a": 9}) == {"a": 9}


def test_merge_or_reset_clears_on_reset():
    assert merge_or_reset({"a": 1}, RESET) == {}


def test_reset_is_none_for_msgpack_serializability():
    """A bespoke sentinel class breaks checkpointing.

    LangGraph serializes input writes into the checkpoint with msgpack, which
    raises `TypeError: Type is not msgpack serializable` on a custom object.
    """
    assert RESET is None


# ---------------------------------------------------------------- new_turn

def test_new_turn_populates_every_eval_field():
    state = new_turn("q", experiment_name="e", run_fingerprint="r", index_fingerprint="i")
    missing = [f for f in EVAL_FIELDS if f not in state]
    assert missing == [], f"EVAL_FIELDS not initialised by new_turn: {missing}"


def test_new_turn_does_not_seed_messages():
    """`messages` is owned by the checkpointer and must keep accumulating."""
    assert "messages" not in new_turn(
        "q", experiment_name="e", run_fingerprint="r", index_fingerprint="i"
    )


def test_new_turn_threads_summary_through():
    state = new_turn(
        "q", experiment_name="e", run_fingerprint="r", index_fingerprint="i",
        summary="earlier turns",
    )
    assert state["summary"] == "earlier turns"


def test_new_turn_ids_are_unique():
    kw = dict(experiment_name="e", run_fingerprint="r", index_fingerprint="i")
    assert new_turn("q", **kw)["turn_id"] != new_turn("q", **kw)["turn_id"]


# ---------------------------------------------------------------- graph

@pytest.fixture
def four_node_app():
    """A stand-in for the real pipeline: route -> retrieve -> generate -> sanitize."""

    def route(_):
        return {
            "route": {"matched": True, "rule_name": "projects", "chunk_types": ["project"]},
            "retrieval_mode": "filtered",
            "timings": {"route": 0.001},
            "trace": ["route"],
        }

    def retrieve(_):
        return {
            "documents": [Document(page_content="x" * 400,
                                   metadata={"chunk_type": "project", "section": "projects", "header": "H"})],
            "context_chars": 400,
            "timings": {"retrieve": 0.42},
            "trace": ["retrieve"],
        }

    def generate(_):
        return {
            "draft_answer": "raw [x](https://evil.example.com)",
            "model_used": "120b",
            "model_attempts": [
                {"model": "20b", "ok": False, "rate_limited": True},
                {"model": "120b", "ok": True, "rate_limited": False},
            ],
            "messages": [AIMessage(content="reply")],
            "timings": {"generate": 1.9},
            "trace": ["generate"],
        }

    def sanitize(_):
        return {
            "answer": "raw",
            "link_audit": {"kept": [], "stripped": ["https://evil.example.com"], "stripped_count": 1},
            "timings": {"sanitize": 0.003},
            "trace": ["sanitize"],
        }

    g = StateGraph(GraphState)
    for name, fn in (("route", route), ("retrieve", retrieve),
                     ("generate", generate), ("sanitize", sanitize)):
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "sanitize")
    g.add_edge("sanitize", END)
    return g.compile(checkpointer=MemorySaver())


def _turn(app, n: int, thread: str = "t1"):
    state = new_turn(f"q{n}", experiment_name="e", run_fingerprint="r", index_fingerprint="i")
    state["messages"] = [HumanMessage(content=f"q{n}")]
    return app.invoke(state, {"configurable": {"thread_id": thread}})


def test_reducers_compose_within_one_turn(four_node_app):
    out = _turn(four_node_app, 1)
    assert out["timings"] == {"route": 0.001, "retrieve": 0.42, "generate": 1.9, "sanitize": 0.003}
    assert out["trace"] == ["route", "retrieve", "generate", "sanitize"]
    assert [a["model"] for a in out["model_attempts"]] == ["20b", "120b"]


@pytest.mark.parametrize("turn", [1, 2, 3, 4])
def test_accumulators_do_not_leak_across_turns(four_node_app, turn):
    """REGRESSION: these were growing with every turn on a checkpointed thread."""
    for n in range(1, turn + 1):
        out = _turn(four_node_app, n)
    assert len(out["model_attempts"]) == 2
    assert len(out["trace"]) == 4
    assert set(out["timings"]) == {"route", "retrieve", "generate", "sanitize"}


def test_messages_still_accumulate_across_turns(four_node_app):
    """The counterpart: history must NOT be reset, or follow-ups break."""
    for n in range(1, 4):
        out = _turn(four_node_app, n)
    assert len(out["messages"]) == 6  # 3 turns x (human + ai)


def test_draft_and_final_answer_are_both_retained(four_node_app):
    """Without the pre-sanitize draft, URL hallucination is unmeasurable."""
    out = _turn(four_node_app, 1)
    assert "https://evil.example.com" in out["draft_answer"]
    assert "https://evil.example.com" not in out["answer"]
    assert out["link_audit"]["stripped"] == ["https://evil.example.com"]


def test_fallback_chain_is_visible_in_state(four_node_app):
    """'primary answered' and 'primary was rate-limited' must be distinguishable."""
    out = _turn(four_node_app, 1)
    assert out["model_attempts"][0]["rate_limited"] is True
    assert out["model_attempts"][-1]["ok"] is True
    assert out["model_used"] == "120b"


def test_every_eval_field_survives_a_full_run(four_node_app):
    out = _turn(four_node_app, 1)
    missing = [f for f in EVAL_FIELDS if f not in out]
    assert missing == [], f"EVAL_FIELDS absent from final state: {missing}"


# ---------------------------------------------------------------- chunk_refs

def test_chunk_refs_records_rank_and_metadata(four_node_app):
    out = _turn(four_node_app, 1)
    refs = chunk_refs(out)
    assert refs == [{"rank": 0, "chunk_id": None, "identity": None,
                     "chunk_type": "project", "section": "projects",
                     "header": "H", "part": None, "chars": 400}]


def test_chunk_refs_carries_the_eval_join_keys(four_node_app):
    """REGRESSION: without chunk_id and identity every retrieval metric scores 0."""
    refs = chunk_refs(_turn(four_node_app, 1))
    assert "chunk_id" in refs[0] and "identity" in refs[0]


def test_chunk_refs_excludes_page_content():
    """Run logs store references, not full text - content would bloat them."""
    state = {"documents": [Document(page_content="secret", metadata={})]}
    assert "secret" not in str(chunk_refs(state))


def test_chunk_refs_handles_empty_state():
    assert chunk_refs({}) == []
