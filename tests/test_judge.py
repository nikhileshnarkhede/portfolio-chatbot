"""
Tests for the LLM judge and its calibration (spec §11 stage 5).

The judge is the least trustworthy instrument in the suite, so the tests focus
on the properties that keep it from doing damage: it must never look confident
when it failed, never block a release before it has been validated, and never
be scored by a statistic that class imbalance can inflate.

Everything runs offline against a scripted judge.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage

from eval import calibration
from eval.metrics import judge as judge_mod
from portfolio_chatbot.config import load_config


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


@pytest.fixture
def cfg():
    return load_config()


class ScriptedJudgeLLM:
    """Returns canned JSON, or raises."""

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, _prompt):
        if isinstance(self.reply, Exception):
            raise self.reply
        return AIMessage(content=self.reply)


@pytest.fixture
def judge_with(monkeypatch, cfg):
    def make(reply):
        monkeypatch.setattr(judge_mod, "build_llm",
                            lambda c, model=None: ScriptedJudgeLLM(reply))
        return judge_mod.build_judge(cfg)
    return make


# ---------------------------------------------------------------- config

def test_judge_defaults_to_the_largest_model(cfg):
    """'The strongest available' - a weak judge produces numbers that look like
    measurements and are not."""
    assert cfg.judge_model == "openai/gpt-oss-120b"


def test_judge_model_can_be_pinned():
    pinned = load_config(overrides=['eval.judge.model="qwen/qwen3.6-27b"'])
    assert pinned.judge_model == "qwen/qwen3.6-27b"


def test_judge_runs_at_zero_temperature(cfg, judge_with):
    """A judge that samples disagrees with itself between runs."""
    j = judge_with('{"score": 8}')
    assert j._judge_cfg.llm.temperature == 0.0
    assert j._judge_cfg.llm.streaming is False


# ---------------------------------------------------------------- parsing

def test_extract_json_handles_a_bare_object():
    assert judge_mod.extract_json('{"score": 7}') == {"score": 7}


def test_extract_json_handles_a_markdown_fence():
    assert judge_mod.extract_json('```json\n{"score": 7}\n```') == {"score": 7}


def test_extract_json_handles_trailing_prose():
    """Models append explanations despite being told not to."""
    assert judge_mod.extract_json('{"score": 7}\n\nI hope this helps!') == {"score": 7}


def test_extract_json_returns_none_on_garbage():
    assert judge_mod.extract_json("no json here") is None


# ---------------------------------------------------------------- scoring

def test_faithfulness_counts_supported_claims(judge_with):
    j = judge_with('{"claims": [{"claim":"a","supported":true},'
                   '{"claim":"b","supported":false},'
                   '{"claim":"c","supported":true}]}')
    assert j.faithfulness("q", "ctx", "answer") == pytest.approx(2 / 3)


def test_faithfulness_collects_the_unsupported_claims(judge_with):
    """When a score drops, the claims are what you actually want to read."""
    j = judge_with('{"claims": [{"claim":"invented fact","supported":false}]}')
    j.faithfulness("q", "ctx", "answer")
    assert "invented fact" in j.unsupported_claims


def test_no_claims_is_vacuously_faithful(judge_with):
    """A refusal makes no factual claims. That is an outcome, not a gap."""
    assert judge_with('{"claims": []}').faithfulness("q", "ctx", "I don't have that.") == 1.0


def test_judge_failure_returns_nan_not_zero(judge_with):
    """A judge that could not be reached must not look like a bad answer."""
    j = judge_with(RuntimeError("judge down"))
    assert is_nan(j.faithfulness("q", "ctx", "answer"))
    assert j.failures == 1


def test_unparseable_response_returns_nan(judge_with):
    assert is_nan(judge_with("I think it's mostly fine").faithfulness("q", "ctx", "a"))


def test_relevancy_normalises_to_zero_one(judge_with):
    assert judge_with('{"score": 8}').relevancy("q", "a") == 0.8


def test_relevancy_clamps_out_of_range_scores(judge_with):
    assert judge_with('{"score": 42}').relevancy("q", "a") == 1.0


def test_empty_answer_is_not_scored(judge_with):
    assert is_nan(judge_with('{"score": 9}').relevancy("q", "   "))


def test_results_are_cached(judge_with):
    j = judge_with('{"score": 8}')
    j.relevancy("q", "a")
    j.relevancy("q", "a")
    assert j.calls == 1, "re-scoring an unchanged answer must cost nothing"


# ---------------------------------------------------------------- kappa

def test_perfect_agreement():
    labels = [True] * 8 + [False] * 8
    assert calibration.cohens_kappa(labels, labels) == 1.0


def test_a_constant_judge_scores_zero_despite_high_raw_agreement():
    """The reason kappa is used instead of raw agreement.

    A judge that answers 'supported' every time agrees with a balanced set 50%
    of the time while carrying no information whatsoever.
    """
    labels = [True] * 8 + [False] * 8
    assert calibration.cohens_kappa(labels, [True] * 16) == 0.0


def test_opposed_raters_score_minus_one():
    labels = [True] * 8 + [False] * 8
    assert calibration.cohens_kappa(labels, [not x for x in labels]) == -1.0


def test_kappa_needs_at_least_two_pairs():
    assert is_nan(calibration.cohens_kappa([True], [True]))


def test_kappa_ignores_unlabelled_pairs():
    assert calibration.cohens_kappa([True, False, None], [True, False, True]) == 1.0


# ---------------------------------------------------------------- dataset

def test_calibration_set_is_balanced():
    """A skewed set lets a constant judge score high agreement."""
    cases = calibration.load_cases()
    supported = sum(1 for c in cases if c["expected_label"])
    assert len(cases) == 30
    assert supported == len(cases) - supported == 15


def test_every_calibration_case_is_complete():
    for case in calibration.load_cases():
        assert case["context"] and case["claim"]
        assert isinstance(case["expected_label"], bool)
        assert "human_label" in case


def test_human_label_takes_precedence_over_authored():
    assert calibration.human_label({"expected_label": True, "human_label": False}) is False
    assert calibration.human_label({"expected_label": True, "human_label": None}) is True


# ---------------------------------------------------------------- verdicts

class FakeJudge:
    """Answers correctly except on every `wrong_every`-th case.

    Deterministic on purpose: a probabilistic fake makes the test flaky, and a
    flaky test about measurement reliability would be its own joke.
    """

    def __init__(self, wrong_every=0, model="fake"):
        self.wrong_every, self.model, self.n = wrong_every, model, 0

    def supports(self, context, claim):
        self.n += 1
        truth = "not" not in claim
        if self.wrong_every and self.n % self.wrong_every == 0:
            return not truth
        return truth


def _cases(n=20):
    return [
        {"id": f"c{i}", "context": "ctx", "claim": "supported" if i % 2 else "not supported",
         "expected_label": bool(i % 2), "human_label": None}
        for i in range(n)
    ]


def test_perfect_judge_earns_trust(cfg):
    result = calibration.calibrate(cfg, FakeJudge(), _cases())
    assert result.kappa == 1.0
    assert result.verdict == "trust"
    assert result.may_block is True


def test_poor_judge_is_suppressed(cfg):
    result = calibration.calibrate(cfg, FakeJudge(wrong_every=2), _cases())
    assert result.verdict in ("suppress", "direction")
    if result.verdict == "suppress":
        assert result.may_block is False


def test_disagreements_are_reported(cfg):
    result = calibration.calibrate(cfg, FakeJudge(wrong_every=2), _cases())
    assert result.disagreements
    assert "claim" in result.disagreements[0]


def test_judge_failures_are_counted_not_scored(cfg):
    class Failing:
        model = "broken"
        def supports(self, *a): return None

    result = calibration.calibrate(cfg, Failing(), _cases())
    assert result.n == 0
    assert result.n_judge_failed == 20
    assert result.verdict == "unmeasured"


def test_label_source_is_recorded(cfg):
    result = calibration.calibrate(cfg, FakeJudge(), _cases())
    assert result.label_source == "expected"


def test_thresholds_come_from_config():
    strict = load_config(overrides=["eval.calibration.min_kappa_trust=0.99"])
    result = calibration.calibrate(strict, FakeJudge(), _cases())
    assert result.verdict == "trust"  # kappa 1.0 still clears 0.99


def test_calibration_thresholds_must_be_ordered():
    with pytest.raises(Exception):
        load_config(overrides=[
            "eval.calibration.min_kappa_trust=0.5",
            "eval.calibration.min_kappa_direction=0.9",
        ])
