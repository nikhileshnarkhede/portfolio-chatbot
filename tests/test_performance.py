"""
Tests for D5 telemetry: percentiles, TTFT, tokens, cost, cold start.

Spec §11 stage 1. These metrics are cheap to compute and easy to get subtly
wrong in ways that never raise - an interpolated p95 that no request
experienced, a TTFT flattered by an empty first chunk, a cost silently computed
at someone else's rate card. Each of those has a test here.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from eval.metrics import performance as perf
from portfolio_chatbot.config import load_config
from portfolio_chatbot.llm.fallback import harvest_usage, run_with_fallback
from portfolio_chatbot.observability import perf as perf_registry

NAN = float("nan")


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------- percentiles

def test_percentile_returns_a_real_observation():
    """Nearest-rank, not interpolation: p95 must be a number that happened."""
    data = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert perf.percentile(data, 95) in data


def test_percentile_matches_nearest_rank():
    data = [10, 20, 30, 40, 50]
    assert perf.percentile(data, 50) == 30
    assert perf.percentile(data, 100) == 50
    assert perf.percentile(data, 1) == 10


def test_percentile_works_with_one_observation():
    """statistics.quantiles needs n>=2. A single slow turn still has a p95."""
    assert perf.percentile([2.5], 95) == 2.5


def test_percentile_of_nothing_is_nan_not_zero():
    """Zero would read as 'instant'. NaN reads as 'not measured'."""
    assert is_nan(perf.percentile([], 95))


def test_percentile_ignores_nan_and_none():
    assert perf.percentile([1.0, None, NAN, 3.0], 100) == 3.0


def test_stdev_needs_two_points():
    assert is_nan(perf.stdev([1.0]))
    assert perf.stdev([2.0, 4.0]) == 1.0


# ---------------------------------------------------------------- TTFT

class Streamer:
    """Stands in for a chat model. `lead_empty` mimics a role-only first chunk."""

    def __init__(self, words, lead_empty=False, usage=None):
        self.words, self.lead_empty, self.usage = words, lead_empty, usage

    def __iter__(self):
        if self.lead_empty:
            yield AIMessageChunk(content="")
        for i, w in enumerate(self.words):
            last = i == len(self.words) - 1
            chunk = AIMessageChunk(content=w)
            if last and self.usage:
                chunk.usage_metadata = self.usage
            yield chunk


@pytest.fixture
def cfg():
    return load_config()


def test_ttft_is_recorded_for_a_stream(cfg):
    result = run_with_fallback(cfg, lambda m: Streamer(["a ", "b ", "c"]))
    assert result.ttft_s is not None
    assert result.ttft_s >= 0


def test_ttft_ignores_an_empty_leading_chunk(cfg):
    """Counting a role-only chunk would flatter TTFT with a packet nobody sees."""
    import time

    def slow_after_empty(_model):
        yield AIMessageChunk(content="")
        time.sleep(0.05)
        yield AIMessageChunk(content="real text")

    result = run_with_fallback(cfg, slow_after_empty)
    assert result.ttft_s >= 0.05, "TTFT must start counting from the first visible token"


def test_ttft_equals_latency_when_not_streaming(cfg):
    """Non-streaming: nothing is visible until the whole answer lands."""
    result = run_with_fallback(cfg, lambda m: "one complete answer")
    assert result.ttft_s is not None
    assert result.attempts[-1]["ttft_s"] == pytest.approx(result.attempts[-1]["latency_s"], abs=0.05)


def test_failed_attempt_has_no_ttft(cfg, rate_limit_error):
    def fn(model):
        if model == cfg.llm.model_chain[0]:
            raise rate_limit_error()
        return "recovered"

    result = run_with_fallback(cfg, fn)
    assert result.attempts[0]["ttft_s"] is None
    assert result.attempts[-1]["ttft_s"] is not None


# ---------------------------------------------------------------- token usage

def test_harvest_usage_reads_usage_metadata():
    chunk = AIMessage(content="x")
    chunk.usage_metadata = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    assert harvest_usage(chunk) == (120, 30)


def test_harvest_usage_returns_none_when_absent():
    """None, not (0,0): 'not reported' and 'no tokens' are different facts."""
    assert harvest_usage(AIMessage(content="x")) is None


def test_usage_flows_through_the_fallback_chain(cfg):
    usage = {"input_tokens": 900, "output_tokens": 100, "total_tokens": 1000}
    result = run_with_fallback(cfg, lambda m: Streamer(["a ", "b"], usage=usage))
    assert result.usage_available is True
    assert result.usage(cfg)["total_tokens"] == 1000


def test_usage_marked_unavailable_when_provider_is_silent(cfg):
    result = run_with_fallback(cfg, lambda m: Streamer(["a ", "b"]))
    assert result.usage(cfg)["available"] is False
    assert result.usage(cfg)["total_tokens"] == 0


# ---------------------------------------------------------------- cost

def test_cost_is_zero_when_unpriced(cfg):
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000}
    result = run_with_fallback(cfg, lambda m: Streamer(["x"], usage=usage))
    assert result.usage(cfg)["cost_usd"] == 0.0


def test_cost_uses_the_configured_rate():
    priced = load_config(overrides=[
        'llm.pricing={"default":{"input_per_1m":10.0,"output_per_1m":30.0}}'
    ])
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000}
    result = run_with_fallback(priced, lambda m: Streamer(["x"], usage=usage))
    assert result.usage(priced)["cost_usd"] == pytest.approx(40.0)


def test_per_model_rate_beats_the_default(cfg):
    priced = load_config(overrides=[
        'llm.pricing={"default":{"input_per_1m":1.0,"output_per_1m":1.0},'
        '"openai/gpt-oss-20b":{"input_per_1m":0.0,"output_per_1m":0.0}}'
    ])
    assert priced.llm.rate_for("openai/gpt-oss-20b") == (0.0, 0.0)
    assert priced.llm.rate_for("something-else") == (1.0, 1.0)


# ---------------------------------------------------------------- aggregation

def _turn(ttft=1.0, latency=2.0, tokens=(100, 50), attempts=1, error=None, timings=None):
    return {
        "ttft_s": ttft,
        "latency_s": latency,
        "token_usage": {"input_tokens": tokens[0], "output_tokens": tokens[1],
                        "total_tokens": sum(tokens), "cost_usd": 0.0, "available": True},
        "model_attempts": [{"model": f"m{i}"} for i in range(attempts)],
        "error": error,
        "timings": timings or {"retrieve": 0.4, "generate": 1.5},
    }


def test_score_aggregates_percentiles():
    turns = [_turn(ttft=t, latency=t * 2) for t in (0.5, 1.0, 1.5, 2.0, 9.0)]
    scores = perf.score(turns)
    assert scores.ttft["p50"] == 1.5
    assert scores.ttft["max"] == 9.0
    assert scores.n_turns == 5


def test_score_reports_per_node_latency():
    """A turn-level p95 cannot say whether the slow tail is retrieval or generation."""
    scores = perf.score([_turn(timings={"retrieve": 0.4, "generate": 3.0})])
    assert set(scores.node_latency) == {"retrieve", "generate"}
    assert scores.node_latency["generate"]["p50"] == 3.0


def test_score_computes_rates():
    turns = [_turn(), _turn(attempts=2), _turn(error="boom"), _turn()]
    scores = perf.score(turns)
    assert scores.fallback_rate == 0.25
    assert scores.error_rate == 0.25


def test_score_totals_tokens():
    scores = perf.score([_turn(tokens=(100, 50)), _turn(tokens=(200, 25))])
    assert scores.input_tokens_total == 300
    assert scores.output_tokens_total == 75
    assert scores.tokens_per_turn_mean == pytest.approx(187.5)


def test_score_of_no_turns_does_not_crash():
    assert perf.score([]).n_turns == 0


def test_render_produces_a_block():
    out = perf.render(perf.score([_turn(), _turn(ttft=3.0)]))
    assert "TTFT" in out and "tokens" in out and "cost" in out


# ---------------------------------------------------------------- cold start

def test_perf_registry_records_first_value_only():
    """These measure COLD start; a second call is a warm path by definition."""
    perf_registry.reset()
    perf_registry.record("index_load", 4.2)
    perf_registry.record("index_load", 0.001)
    assert perf_registry.get("index_load") == 4.2
    perf_registry.reset()


def test_perf_registry_totals():
    perf_registry.reset()
    perf_registry.record("embedder_init", 3.0)
    perf_registry.record("index_load", 1.0)
    assert perf_registry.snapshot()["cold_start_total"] == 4.0
    perf_registry.reset()


def test_timed_context_manager_records_on_exception():
    perf_registry.reset()
    with pytest.raises(ValueError):
        with perf_registry.timed("boom"):
            raise ValueError("x")
    assert perf_registry.get("boom") is not None
    perf_registry.reset()


def test_empty_registry_snapshot_has_no_total():
    perf_registry.reset()
    assert perf_registry.snapshot() == {}


# ------------------------------------------------- regression: content extraction

def test_content_of_extracts_text_not_repr():
    """REGRESSION: str(chunk) yields the object's repr.

    A naive fallback accumulated "content='' additional_kwargs={}..." straight
    into the answer. Caught only because a TTFT test happened to print the
    buffer.
    """
    from portfolio_chatbot.llm.fallback import content_of
    assert content_of(AIMessageChunk(content="hello")) == "hello"
    assert content_of("hello") == "hello"
    assert "additional_kwargs" not in content_of(AIMessageChunk(content=""))


def test_content_of_handles_block_style_content():
    from portfolio_chatbot.llm.fallback import content_of
    msg = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert content_of(msg) == "ab"


def test_usage_survives_a_non_streaming_message(cfg):
    """REGRESSION: generate.py used to strip .content before the fallback saw it,
    so usage_metadata was discarded and usage_reported would have been 0 forever."""
    msg = AIMessage(content="an answer")
    msg.usage_metadata = {"input_tokens": 42, "output_tokens": 8, "total_tokens": 50}
    result = run_with_fallback(cfg, lambda m: msg)
    assert result.text == "an answer"
    assert result.usage(cfg)["total_tokens"] == 50
    assert result.usage(cfg)["available"] is True
