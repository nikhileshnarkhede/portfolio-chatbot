"""
Tests for the individual nodes and the tools underneath them.

Everything here runs offline: a real FAISS index built with a deterministic
fake embedder, and a scripted chat model that can raise rate-limit errors on
demand. That is deliberate - the fallback chain and the link guard are exactly
the paths that are hard to reach in a browser and easy to break silently.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from portfolio_chatbot.config import load_config
from portfolio_chatbot.llm.fallback import (
    AllModelsFailed,
    is_rate_limited,
    run_with_fallback,
)
from portfolio_chatbot.memory import history
from portfolio_chatbot.nodes.expand_query import expand_query
from portfolio_chatbot.nodes.generate import RATE_LIMITED_REPLY, generate
from portfolio_chatbot.nodes.retrieve import retrieve
from portfolio_chatbot.nodes.route import route
from portfolio_chatbot.nodes.summarize import summarize
from portfolio_chatbot.state import new_turn
from portfolio_chatbot.tools import retriever_tool

ALLOWED = "https://github.com/nikhileshnarkhede"
FORGED = "https://github.com/nikhileshnarkhede-fake"


def state_for(question: str, **extra):
    s = new_turn(question, experiment_name="t", run_fingerprint="r", index_fingerprint="i")
    s.update(extra)
    return s


# ---------------------------------------------------------------- node plumbing

def test_nodes_reject_a_missing_config():
    """A node without an AppConfig must say so, not fail three frames deeper."""
    with pytest.raises(KeyError, match="app_config"):
        route(state_for("hi"), {})


# ---------------------------------------------------------------- route

@pytest.mark.parametrize("question,expected_rule", [
    ("What are all your technical skills?", "skills"),
    ("Tell me about your projects", "projects"),
    ("Where do you work currently?", "current_role"),
    ("Tell me about your PINN research", "research"),
    ("What degree do you have?", "education"),
    ("What certifications do you hold?", "certifications"),
])
def test_route_matches_expected_rule(question, expected_rule, runnable_config):
    assert route(state_for(question), runnable_config)["route"]["rule_name"] == expected_rule


def test_route_falls_through_on_an_unmatched_question(runnable_config):
    decision = route(state_for("What is your favourite colour?"), runnable_config)["route"]
    assert decision["matched"] is False
    assert decision["chunk_types"] == []


def test_route_order_puts_current_role_before_generic_job(runnable_config):
    """'current job' contains 'job'; rule order is what keeps them apart."""
    assert route(state_for("what is your current job"), runnable_config)["route"]["rule_name"] == "current_role"
    assert route(state_for("what was your first job"), runnable_config)["route"]["rule_name"] == "experience"


def test_route_records_the_matched_keyword(runnable_config):
    assert route(state_for("tell me about your projects"), runnable_config)["route"]["matched_keyword"] == "project"


def test_route_can_be_disabled_by_config():
    cfg = load_config(overrides=["retrieval.routing.enabled=false"])
    out = route(state_for("skills"), {"configurable": {"app_config": cfg}})
    assert out["route"]["matched"] is False


def test_route_reads_the_raw_question_not_the_expanded_one(runnable_config):
    """Expansion appends 'projects applications...' - routing must ignore it."""
    s = state_for("what university did you attend",
                  expanded_question="what university did you attend projects applications systems")
    assert route(s, runnable_config)["route"]["rule_name"] == "education"


# ---------------------------------------------------------------- expand_query

def test_keyword_expansion_appends_terms(runnable_config):
    out = expand_query(state_for("what are your skills"), runnable_config)
    assert out["expanded_question"].startswith("what are your skills")
    assert len(out["expanded_question"]) > len("what are your skills")


def test_expansion_leaves_unmatched_questions_alone(runnable_config):
    q = "what is your favourite colour"
    assert expand_query(state_for(q), runnable_config)["expanded_question"] == q


def test_expansion_can_be_disabled():
    cfg = load_config(overrides=["retrieval.query_expansion.enabled=false"])
    q = "what are your skills"
    out = expand_query(state_for(q), {"configurable": {"app_config": cfg}})
    assert out["expanded_question"] == q


def test_llm_expansion_falls_back_to_the_raw_question_on_failure(scripted_llm, rate_limit_error):
    """A failed rewrite must degrade, not kill the turn."""
    scripted_llm({"*": rate_limit_error()})
    cfg = load_config(overrides=['retrieval.query_expansion.mode="llm"'])
    q = "tell me more"
    out = expand_query(state_for(q), {"configurable": {"app_config": cfg}})
    assert out["expanded_question"] == q


# ---------------------------------------------------------------- retrieve

def test_retrieve_uses_the_filter_when_routing_matched(cfg, fake_index, runnable_config):
    s = state_for("tell me about your projects", expanded_question="projects",
                  route={"matched": True, "chunk_types": ["project"]})
    out = retrieve(s, runnable_config)
    assert out["retrieval_mode"] == "filtered"
    assert {d.metadata["chunk_type"] for d in out["documents"]} == {"project"}


def test_retrieve_falls_back_to_mmr_without_a_route(cfg, fake_index, runnable_config):
    out = retrieve(state_for("anything", expanded_question="anything"), runnable_config)
    assert out["retrieval_mode"] == "mmr"
    assert out["documents"]


def test_retrieve_falls_back_when_the_filter_matches_nothing(cfg, fake_index, runnable_config):
    s = state_for("q", expanded_question="q",
                  route={"matched": True, "chunk_types": ["no_such_type"]})
    assert retrieve(s, runnable_config)["retrieval_mode"] == "mmr"


def test_context_is_truncated_and_flagged(cfg, fake_index):
    small = load_config(overrides=["retrieval.context.max_context_chars=200"])
    out = retrieve(state_for("projects", expanded_question="projects"),
                   {"configurable": {"app_config": small}})
    assert out["context_truncated"] is True
    assert out["context_chars"] == 200


def test_missing_index_raises_a_useful_message(cfg):
    retriever_tool.clear_cache()
    with pytest.raises(FileNotFoundError, match="scripts/ingest.py"):
        retriever_tool.search(load_config(overrides=['paths.index_root="data/nope"']), "q", None)


def test_empty_results_produce_the_configured_message(cfg):
    context, truncated = retriever_tool.format_context([], cfg)
    assert context == cfg.retrieval.context.empty_message
    assert truncated is False


# ---------------------------------------------------------------- fallback chain

def test_is_rate_limited_detects_common_shapes():
    assert is_rate_limited("Error code: 429")
    assert is_rate_limited("rate_limit_exceeded")
    assert not is_rate_limited("invalid api key")


def test_fallback_skips_past_rate_limited_models(cfg, rate_limit_error):
    calls = []

    def fn(model):
        calls.append(model)
        if model != cfg.llm.model_chain[2]:
            raise rate_limit_error()
        return "answer"

    result = run_with_fallback(cfg, fn)
    assert result.text == "answer"
    assert result.model_used == cfg.llm.model_chain[2]
    assert len(result.attempts) == 3
    assert [a["ok"] for a in result.attempts] == [False, False, True]


def test_fallback_does_not_retry_on_a_non_rate_limit_error(cfg):
    calls = []

    def fn(model):
        calls.append(model)
        raise ValueError("invalid api key")

    with pytest.raises(AllModelsFailed):
        run_with_fallback(cfg, fn)
    assert len(calls) == 1, "a bad key must not burn the whole chain"


def test_fallback_remembers_where_to_start(cfg, rate_limit_error):
    def fn(model):
        if model == cfg.llm.model_chain[0]:
            raise rate_limit_error()
        return "ok"

    first = run_with_fallback(cfg, fn)
    assert first.next_start_index == 1

    second = run_with_fallback(cfg, fn, start_index=first.next_start_index)
    assert len(second.attempts) == 1, "should not re-hit the exhausted primary"


def test_fallback_can_be_disabled(cfg, rate_limit_error):
    single = load_config(overrides=["llm.fallback_on_rate_limit=false"])
    with pytest.raises(AllModelsFailed) as exc:
        run_with_fallback(single, lambda m: (_ for _ in ()).throw(rate_limit_error()))
    assert len(exc.value.attempts) == 1


def test_streaming_chunks_reach_the_callback(cfg):
    seen = []
    run_with_fallback(cfg, lambda m: iter(["a", "b", "c"]), on_chunk=seen.append)
    assert seen == ["a", "b", "c"]


# ---------------------------------------------------------------- generate

def test_generate_writes_draft_not_answer(scripted_llm, runnable_config):
    scripted_llm({"*": "hello there"})
    out = generate(state_for("hi", context="CTX"), runnable_config)
    assert out["draft_answer"].strip() == "hello there"
    assert "answer" not in out, "sanitize owns `answer`"


def test_generate_records_the_fallback_it_took(scripted_llm, runnable_config, cfg, rate_limit_error):
    scripted_llm({cfg.llm.model_chain[0]: rate_limit_error(), "*": "recovered"})
    out = generate(state_for("hi", context="CTX"), runnable_config)
    assert out["model_used"] == cfg.llm.model_chain[1]
    assert out["model_attempts"][0]["rate_limited"] is True


def test_generate_survives_total_failure(scripted_llm, runnable_config, rate_limit_error):
    """An eval run of 100 questions must not abort because the free tier ran dry."""
    scripted_llm({"*": rate_limit_error()})
    out = generate(state_for("hi", context="CTX"), runnable_config)
    assert out["draft_answer"] == RATE_LIMITED_REPLY
    assert out["error"]
    assert len(out["model_attempts"]) == 4


# ---------------------------------------------------------------- history

def test_history_uses_the_configured_speaker_labels(cfg):
    text = history.transcript([HumanMessage(content="hi"), AIMessage(content="hello")], cfg)
    assert text == "Recruiter: hi\nNikhilesh: hello"


def test_history_reports_no_conversation_when_empty(cfg):
    assert history.for_prompt([], "", cfg) == history.NO_HISTORY


def test_history_includes_the_summary_when_present(cfg):
    text = history.for_prompt([HumanMessage(content="hi")], "earlier stuff", cfg)
    assert "earlier stuff" in text and "Recruiter: hi" in text


# ---------------------------------------------------------------- summarize

def _conversation(n: int):
    msgs = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"q{i}", id=f"h{i}"))
        msgs.append(AIMessage(content=f"a{i}", id=f"a{i}"))
    return msgs


def test_summarize_is_a_noop_below_the_threshold(scripted_llm, runnable_config):
    scripted_llm({"*": "summary"})
    assert summarize(state_for("q", messages=_conversation(2)), runnable_config) == {}


def test_summarize_trims_and_summarizes_above_the_threshold(scripted_llm, runnable_config, cfg):
    scripted_llm({"*": "they discussed projects"})
    msgs = _conversation(8)  # 16 messages, threshold is 12
    out = summarize(state_for("q", messages=msgs), runnable_config)
    assert out["summarized_this_turn"] is True
    assert out["summary"] == "they discussed projects"
    assert len(out["messages"]) == len(msgs) - cfg.memory.keep_last_n


def test_summarize_rolls_the_previous_summary_forward(monkeypatch, runnable_config):
    """REGRESSION vs the original: the old code discarded the earlier summary."""
    captured = {}

    class Capturing:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return AIMessage(content="merged summary")

    # monkeypatch, not a bare assignment: a leaked patch here would silently
    # become the "original" that later fixtures restore to.
    monkeypatch.setattr("portfolio_chatbot.nodes.summarize.build_llm",
                        lambda cfg, model=None: Capturing())

    out = summarize(state_for("q", messages=_conversation(8), summary="EARLIER FACTS"),
                    runnable_config)
    assert "EARLIER FACTS" in captured["prompt"]
    assert out["summary"] == "merged summary"


def test_summarize_still_trims_when_the_model_fails(scripted_llm, runnable_config, rate_limit_error):
    """History is too long - that problem does not go away if the LLM is down."""
    scripted_llm({"*": rate_limit_error()})
    out = summarize(state_for("q", messages=_conversation(8), summary="kept"), runnable_config)
    assert out["messages"], "messages must still be trimmed"
    assert out["summary"] == "kept"
