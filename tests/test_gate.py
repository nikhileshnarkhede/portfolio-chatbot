"""
Tests for the regression gate and the noise floor (spec §11 stage 4).

The gate is the component most likely to be quietly disabled, so its behaviour
has to be defensible: it must fire on real movement, stay silent on noise, and
never block on a metric it did not actually measure.

The `requires_kappa` tests encode the rule from spec §07 that an unvalidated
judge cannot stop a release. Without it, `faithfulness` - a number from an
uncalibrated instrument - would have the power to block a deploy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval import gate, noise


@dataclass
class FakeReport:
    summary: dict
    dataset: str = "golden_qa"
    experiment: str = "default"
    run_fingerprint: str = "abc123"
    index_fingerprint: str = "def456"
    n_turns: int = 22
    n_errors: int = 0


SPEC = {
    "invariants": {
        "leaked_url_count": {"max": 0, "why": "guard failed"},
        "fabricated_entity_rate": {"max": 0, "why": "invented fact"},
        "human_rating_min": {"min": 3, "why": "harmful answer"},
    },
    "thresholds": {
        "entity_grounding": {"min": 0.98, "on_fail": "block"},
        "hit_at_k": {"min": 0.95, "on_fail": "warn"},
        "faithfulness": {"min": 0.95, "on_fail": "block", "requires_kappa": 0.75},
        "error_rate": {"max": 0.02, "on_fail": "block"},
    },
    "regression": {"sigma_multiple": 2.0, "min_absolute": 0.02},
}

CLEAN = {
    "leaked_url_count": 0, "fabricated_entity_rate": 0.0,
    "entity_grounding": 1.0, "hit_at_k": 1.0, "error_rate": 0.0,
}


# ---------------------------------------------------------------- invariants

def test_clean_report_passes():
    result = gate.check(FakeReport(CLEAN), spec=SPEC)
    assert result.passed and result.clean


def test_leaked_url_blocks_unconditionally():
    """No baseline, no tolerance. The guard failing is not a metric regression."""
    result = gate.check(FakeReport({**CLEAN, "leaked_url_count": 1}), spec=SPEC)
    assert not result.passed
    assert result.blocking[0].kind == "invariant"


def test_fabricated_entity_blocks():
    result = gate.check(FakeReport({**CLEAN, "fabricated_entity_rate": 0.05}), spec=SPEC)
    assert not result.passed


def test_invariant_with_a_minimum_blocks_below_it():
    result = gate.check(FakeReport({**CLEAN, "human_rating_min": 2}), spec=SPEC)
    assert not result.passed
    assert any(v.metric == "human_rating_min" for v in result.blocking)


def test_unmeasured_invariant_is_not_a_failure():
    """A suite that does not produce human ratings must not fail on them."""
    assert gate.check(FakeReport(CLEAN), spec=SPEC).passed


# ---------------------------------------------------------------- thresholds

def test_threshold_below_minimum_blocks():
    result = gate.check(FakeReport({**CLEAN, "entity_grounding": 0.90}), spec=SPEC)
    assert not result.passed


def test_warn_threshold_does_not_block():
    result = gate.check(FakeReport({**CLEAN, "hit_at_k": 0.50}), spec=SPEC)
    assert result.passed, "a warn-level failure must not block"
    assert not result.clean
    assert len(result.warnings) == 1


def test_maximum_threshold_blocks_when_exceeded():
    result = gate.check(FakeReport({**CLEAN, "error_rate": 0.5}), spec=SPEC)
    assert not result.passed


def test_unmeasured_thresholds_are_reported_as_skipped():
    result = gate.check(FakeReport({"leaked_url_count": 0}), spec=SPEC)
    assert "entity_grounding" in result.skipped


# ---------------------------------------------------------------- kappa gating

def test_judged_metric_cannot_block_without_a_validated_judge():
    """An unvalidated instrument must not have the power to stop a release."""
    failing = {**CLEAN, "faithfulness": 0.40}
    result = gate.check(FakeReport(failing), spec=SPEC, kappa=None)
    assert result.passed, "faithfulness must demote to warn without kappa"
    assert any(v.metric == "faithfulness" for v in result.warnings)


def test_judged_metric_blocks_once_kappa_is_high_enough():
    failing = {**CLEAN, "faithfulness": 0.40}
    result = gate.check(FakeReport(failing), spec=SPEC, kappa=0.81)
    assert not result.passed


def test_low_kappa_still_demotes():
    failing = {**CLEAN, "faithfulness": 0.40}
    assert gate.check(FakeReport(failing), spec=SPEC, kappa=0.55).passed


# ---------------------------------------------------------------- regression

def _baseline(summary, sigma=None):
    return {"dataset": "golden_qa", "run_fingerprint": "base1234",
            "summary": summary, "sigma": sigma or {}}


def test_regression_beyond_two_sigma_fires():
    base = _baseline({"entity_grounding": 1.0}, {"entity_grounding": 0.01})
    result = gate.check(FakeReport({**CLEAN, "entity_grounding": 0.90}), base, SPEC)
    assert any(v.kind == "regression" for v in result.violations)


def test_movement_within_the_noise_floor_does_not_fire():
    """The whole point of measuring sigma: do not gate on wobble."""
    base = _baseline({"entity_grounding": 1.0}, {"entity_grounding": 0.05})
    result = gate.check(FakeReport({**CLEAN, "entity_grounding": 0.95}), base, SPEC)
    assert not any(v.kind == "regression" for v in result.violations)


def test_missing_sigma_falls_back_to_min_absolute():
    base = _baseline({"entity_grounding": 1.0})  # no sigma recorded
    result = gate.check(FakeReport({**CLEAN, "entity_grounding": 0.90}), base, SPEC)
    regressions = [v for v in result.violations if v.kind == "regression"]
    assert regressions and "no measured sigma" in regressions[0].message


def test_improvement_never_counts_as_a_regression():
    base = _baseline({"entity_grounding": 0.80}, {"entity_grounding": 0.01})
    result = gate.check(FakeReport({**CLEAN, "entity_grounding": 1.0}), base, SPEC)
    assert not any(v.kind == "regression" for v in result.violations)


def test_direction_is_respected_for_lower_is_better():
    base = _baseline({"error_rate": 0.0}, {"error_rate": 0.001})
    result = gate.check(FakeReport({**CLEAN, "error_rate": 0.0}), base, SPEC)
    assert result.passed
    worse = gate.check(FakeReport({**CLEAN, "error_rate": 0.3}), base, SPEC)
    assert any(v.kind == "regression" for v in worse.violations)


# ---------------------------------------------------------------- persistence

def test_save_and_load_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "BASELINE_DIR", tmp_path)
    report = FakeReport({**CLEAN, "hit_at_k": 0.8})
    gate.save_baseline(report, sigmas={"hit_at_k": 0.02}, n_repeats=5)
    loaded = gate.load_baseline("golden_qa")
    assert loaded["summary"]["hit_at_k"] == 0.8
    assert loaded["sigma"]["hit_at_k"] == 0.02
    assert loaded["_meta"]["sigma_measured"] is True


def test_baseline_drops_non_numeric_summary_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "BASELINE_DIR", tmp_path)
    gate.save_baseline(FakeReport({**CLEAN, "modes": {"mmr": 3}, "ungrounded_entities": ["x"]}))
    loaded = gate.load_baseline("golden_qa")
    assert "modes" not in loaded["summary"]


def test_shipped_thresholds_file_is_valid():
    spec = gate.load_thresholds()
    assert spec["invariants"] and spec["thresholds"]
    for metric, rule in spec["thresholds"].items():
        assert rule["on_fail"] in ("block", "warn"), metric
        assert ("min" in rule) or ("max" in rule), metric


# ---------------------------------------------------------------- noise floor

class Rep:
    def __init__(self, summary): self.summary = summary


def test_noise_floor_computes_sigma_and_mde():
    values = [0.80, 0.82, 0.78, 0.81, 0.79]
    floor = noise.measure(lambda i: Rep({"hit_at_k": values[i]}), repeats=5)
    assert floor.sigmas["hit_at_k"] > 0
    assert floor.mde["hit_at_k"] == pytest.approx(floor.sigmas["hit_at_k"] * 2, abs=1e-6)


def test_deterministic_metric_has_zero_sigma():
    """Retrieval involves no sampling; its sigma should be exactly 0."""
    floor = noise.measure(lambda i: Rep({"route_correct": 0.82}), repeats=5)
    assert floor.sigmas["route_correct"] == 0.0


def test_is_significant_uses_the_mde():
    floor = noise.measure(lambda i: Rep({"hit_at_k": [0.80, 0.90][i % 2]}), repeats=4)
    assert not floor.is_significant("hit_at_k", 0.01)
    assert floor.is_significant("hit_at_k", 0.5)


def test_unmeasured_metric_is_never_significant():
    floor = noise.measure(lambda i: Rep({"hit_at_k": 0.8}), repeats=3)
    assert not floor.is_significant("faithfulness", 0.9)


def test_noise_floor_requires_at_least_two_runs():
    with pytest.raises(ValueError, match="at least 2"):
        noise.measure(lambda i: Rep({}), repeats=1)


def test_render_flags_very_noisy_metrics():
    floor = noise.measure(lambda i: Rep({"faithfulness": [0.4, 0.9][i % 2]}), repeats=4)
    assert "VERY NOISY" in floor.render()
