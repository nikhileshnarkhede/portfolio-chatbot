"""
Tests for the ratings store, the apply flow, and the dashboard (§11 stages 7-9).

The rule under test throughout: **you cannot apply a config you have not
measured.** Apply is the only thing here that changes production, so every path
into it is checked - including the override, which exists but has to be written
down rather than clicked.

The ratings tests focus on the property that makes a 20-question pass
sustainable: a rating is keyed by the answer's hash, so an unchanged answer
keeps its score across runs and a second pass only asks about what changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from eval import apply as apply_mod
from eval import ratings as ratings_mod
from portfolio_chatbot.config import load_config
from ui.dash import tunables


@pytest.fixture
def ratings_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ratings_mod, "RATINGS_DIR", tmp_path)
    monkeypatch.setattr(ratings_mod, "FIELD_RATINGS", tmp_path / "field.json")
    return tmp_path


# ---------------------------------------------------------------- ratings

def test_rating_round_trips(ratings_dir):
    ratings_mod.save_rating("fp1", "case_a", "an answer", 8, "solid")
    found = ratings_mod.get_rating("fp1", "case_a", "an answer")
    assert found["score"] == 8 and found["note"] == "solid"


def test_rating_is_keyed_by_answer_not_case(ratings_dir):
    """Changing the answer must invalidate the rating; the case id alone must not."""
    ratings_mod.save_rating("fp1", "case_a", "original answer", 9)
    assert ratings_mod.get_rating("fp1", "case_a", "a different answer") is None


def test_unchanged_answer_keeps_its_rating_across_runs(ratings_dir):
    """The property that makes re-rating sustainable."""
    ratings_mod.save_rating("fp_old", "case_a", "same answer", 7)
    assert ratings_mod.find_rating_anywhere("case_a", "same answer")["score"] == 7


def test_unrated_only_returns_what_actually_changed(ratings_dir):
    ratings_mod.save_rating("fp1", "a", "answer one", 8)
    turns = [{"case_id": "a", "answer": "answer one"},
             {"case_id": "b", "answer": "answer two"}]
    assert [t["case_id"] for t in ratings_mod.unrated(turns, "fp1")] == ["b"]


def test_out_of_range_scores_are_rejected(ratings_dir):
    with pytest.raises(ValueError):
        ratings_mod.save_rating("fp1", "a", "answer", 11)


def test_stats_expose_the_worst_answer(ratings_dir):
    """A single 2 matters more than a good mean."""
    for i, score in enumerate([9, 8, 2]):
        ratings_mod.save_rating("fp1", f"c{i}", f"answer {i}", score)
    stats = ratings_mod.stats("fp1")
    assert stats.min == 2 and stats.below_three == ["c2"]


def test_summary_fields_feed_the_gate(ratings_dir):
    ratings_mod.save_rating("fp1", "a", "answer", 6)
    fields = ratings_mod.summary_fields("fp1")
    assert fields["human_rating_mean"] == 6 and fields["human_rating_min"] == 6


def test_rubric_anchors_every_score():
    assert all(ratings_mod.rubric_for(s) for s in range(0, 11))


def test_field_ratings_are_stored_separately(ratings_dir):
    """Not blind, so they must never reach the gate's rating store."""
    ratings_mod.save_field_rating("q", "an answer", 3)
    assert ratings_mod.load_field_ratings()[0]["score"] == 3
    assert ratings_mod.stats("fp1").n == 0


def test_low_field_ratings_surface_as_reference_candidates(ratings_dir):
    ratings_mod.save_field_rating("bad question", "poor answer", 2)
    ratings_mod.save_field_rating("fine question", "good answer", 9)
    assert [c["score"] for c in ratings_mod.field_candidates()] == [2]


# ---------------------------------------------------------------- apply

@dataclass
class FakeReport:
    run_fingerprint: str
    dataset: str = "golden_qa"
    n_turns: int = 40
    summary: dict = field(default_factory=lambda: {"hit_at_k": 0.9})


@dataclass
class FakeGate:
    passed: bool = True
    blocking: list = field(default_factory=list)

    def as_dict(self):
        return {"passed": self.passed}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An isolated configs/ tree so apply never touches the real repo."""
    base = {"_meta": {"keep": "me"}, "retrieval": {"filtered": {"k": 14}},
            "llm": {"temperature": 0.3}}
    default = tmp_path / "default.json"
    default.write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(apply_mod, "DEFAULT_CONFIG_PATH", default)
    monkeypatch.setattr(apply_mod, "EXPERIMENT_DIR", tmp_path / "experiments")
    monkeypatch.setattr(apply_mod, "HISTORY_DIR", tmp_path / "history")
    monkeypatch.setattr(apply_mod, "PROVENANCE_PATH", tmp_path / "provenance.json")
    monkeypatch.setattr(apply_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path, base, default


def test_delta_contains_only_what_changed():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    new = {"a": {"b": 99, "c": 2}, "d": 3}
    assert apply_mod.compute_delta(new, base) == {"a": {"b": 99}}


def test_delta_is_empty_when_nothing_changed():
    base = {"a": 1}
    assert apply_mod.compute_delta(dict(base), base) == {}


def test_apply_is_refused_without_an_evaluation(sandbox):
    cfg = load_config()
    with pytest.raises(apply_mod.ApplyRefused, match="No evaluation"):
        apply_mod.apply(cfg, {}, None, None)


def test_apply_is_refused_when_the_report_is_for_another_config(sandbox):
    cfg = load_config()
    stale = FakeReport(run_fingerprint="not-this-config")
    with pytest.raises(apply_mod.ApplyRefused, match="different config"):
        apply_mod.apply(cfg, {}, stale, FakeGate())


def test_apply_is_refused_when_the_gate_failed(sandbox):
    cfg = load_config()
    report = FakeReport(run_fingerprint=cfg.run_fingerprint)

    @dataclass
    class V:
        metric: str = "entity_grounding"

    with pytest.raises(apply_mod.ApplyRefused, match="gate failed"):
        apply_mod.apply(cfg, {}, report, FakeGate(passed=False, blocking=[V()]))


def test_override_requires_words_not_a_flag(sandbox):
    """An override you have to write down is one you have to actually make.

    Whitespace is not an explanation, so it must not unlock a real failure.
    """
    cfg = load_config()

    @dataclass
    class V:
        metric: str = "entity_grounding"

    failed = FakeGate(passed=False, blocking=[V()])
    report = FakeReport(run_fingerprint=cfg.run_fingerprint)

    with pytest.raises(apply_mod.ApplyRefused):
        apply_mod.apply(cfg, {}, report, failed, override_note="   ")


def test_override_with_a_reason_is_allowed_and_recorded(sandbox):
    """The escape hatch exists - a gate you cannot override gets worked around
    by editing the baseline file, which is strictly worse."""
    tmp_path, base, _ = sandbox
    cfg = load_config()

    @dataclass
    class V:
        metric: str = "entity_grounding"

    result = apply_mod.apply(
        cfg, base, FakeReport(run_fingerprint=cfg.run_fingerprint),
        FakeGate(passed=False, blocking=[V()]),
        override_note="known false positive on the ARDE acronym",
    )
    assert result.overridden is True
    assert apply_mod.load_provenance()[0]["override_note"]


def test_successful_apply_overwrites_archives_and_records(sandbox, existing_index):
    tmp_path, base, default = sandbox
    cfg = load_config()
    report = FakeReport(run_fingerprint=cfg.run_fingerprint)
    new_config = {**base, "retrieval": {"filtered": {"k": 20}}}

    result = apply_mod.apply(cfg, new_config, report, FakeGate(), label="wider k")

    live = json.loads(default.read_text())
    assert live["retrieval"]["filtered"]["k"] == 20
    assert live["_meta"]["keep"] == "me", "the _meta rules block must survive"

    archived = json.loads((tmp_path / "history" / Path_name(result.archived_to)).read_text())
    assert archived["retrieval"]["filtered"]["k"] == 14

    delta = json.loads((tmp_path / "experiments" / f"{result.experiment_name}.json").read_text())
    assert delta["retrieval"]["filtered"]["k"] == 20
    assert "llm" not in delta, "the delta must contain only what changed"

    assert apply_mod.load_provenance()[0]["run_fingerprint"] == cfg.run_fingerprint


def Path_name(path_str: str) -> str:
    return path_str.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def test_rollback_restores_and_archives_the_current(sandbox, existing_index):
    tmp_path, base, default = sandbox
    cfg = load_config()
    report = FakeReport(run_fingerprint=cfg.run_fingerprint)
    apply_mod.apply(cfg, {**base, "llm": {"temperature": 0.9}}, report, FakeGate())
    assert json.loads(default.read_text())["llm"]["temperature"] == 0.9

    apply_mod.rollback(apply_mod.list_history()[-1])
    assert json.loads(default.read_text())["llm"]["temperature"] == 0.3


def test_stale_experiments_are_detected(sandbox):
    tmp_path, base, _ = sandbox
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    (experiments / "exp_k20.json").write_text(
        json.dumps({"retrieval": {"filtered": {"k": 20}}}), encoding="utf-8")

    new_default = {**base, "retrieval": {"filtered": {"k": 20}}}
    assert "exp_k20" in apply_mod.stale_experiments(new_default)


def test_experiment_names_increment(sandbox):
    tmp_path, _, _ = sandbox
    experiments = tmp_path / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    (experiments / "exp007_thing.json").write_text("{}", encoding="utf-8")
    assert apply_mod.next_experiment_name("next one").startswith("exp008_")


# ---------------------------------------------------------------- tunables

def test_every_knob_points_at_a_real_config_path():
    """A typo'd path would silently render a widget that changes nothing."""
    import json as _json
    from portfolio_chatbot.config import DEFAULT_CONFIG_PATH
    live = _json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for knob in tunables.KNOBS:
        assert tunables.get_value(live, knob.path) is not None, knob.path


def test_set_value_does_not_mutate_the_original():
    base = {"a": {"b": 1}}
    updated = tunables.set_value(base, "a.b", 2)
    assert base["a"]["b"] == 1 and updated["a"]["b"] == 2


def test_changed_knobs_reports_before_and_after():
    base = {"llm": {"temperature": 0.3}}
    now = {"llm": {"temperature": 0.9}}
    changes = tunables.changed_knobs(now, base)
    assert any(k.path == "llm.temperature" and was == 0.3 and new == 0.9
               for k, was, new in changes)


def test_chunking_knobs_are_flagged_as_index_rebuilding():
    """Scoring a new chunking against the old index would measure the old chunking."""
    base = {"ingestion": {"split": {"max_chunk_chars": 1300}}}
    now = {"ingestion": {"split": {"max_chunk_chars": 512}}}
    assert tunables.requires_reingest(now, base)


def test_retrieval_knobs_do_not_require_reingest():
    base = {"retrieval": {"filtered": {"k": 14}}}
    now = {"retrieval": {"filtered": {"k": 20}}}
    assert not tunables.requires_reingest(now, base)


def test_select_knob_options_are_valid_for_the_config():
    """An option the config would reject makes the widget a trap."""
    for knob in tunables.KNOBS:
        if knob.kind != "select":
            continue
        for option in knob.options or ():
            value = f'"{option}"' if isinstance(option, str) else option
            load_config(overrides=[f"{knob.path}={value}"])


# ---------------------------------------------------- deploy: a shared image

def test_a_different_resume_gets_a_different_index(tmp_path):
    """The index fingerprint must cover the resume's CONTENT, not its path.

    This is the bug that makes a shareable image dangerous. Somebody mounts
    their own resume over `data/raw/resume.txt`, the path is unchanged, so the
    fingerprint is unchanged, so the container finds the index that was baked
    from SOMEBODY ELSE'S resume sitting exactly where it expects one - and
    answers every question out of it. No error, no warning, no way to notice
    from the outside.
    """
    from portfolio_chatbot.config import load_config

    resume = load_config().resume_path
    original = resume.read_bytes()
    before = load_config().index_fingerprint

    try:
        resume.write_bytes(original.replace(b"Nikhilesh", b"Someone Else", 1))
        assert load_config().index_fingerprint != before
    finally:
        resume.write_bytes(original)

    assert load_config().index_fingerprint == before, \
        "restoring the resume must restore the fingerprint - the hash has to " \
        "be of content, not of when it was last touched"


def test_a_missing_resume_still_yields_a_fingerprint(tmp_path, monkeypatch):
    """`index_path` is read while deciding whether an ingest is needed.

    That question gets asked before the resume is guaranteed to exist - the
    entrypoint asks it on a container whose resume mount could be wrong - so
    hashing must degrade to a value, never to an exception.
    """
    from portfolio_chatbot.config import _digest_file
    assert _digest_file(tmp_path / "nope.txt") == "absent"


def test_entrypoint_ingests_before_starting_the_app():
    """The image is only shareable if it indexes whatever resume it is given."""
    from portfolio_chatbot.config import PROJECT_ROOT
    entrypoint = (PROJECT_ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "scripts/ingest.py" in entrypoint
    assert 'exec "$@"' in entrypoint, "must exec so the app gets PID 1's signals"


def test_compose_never_bakes_the_key_or_pulls_the_image():
    from portfolio_chatbot.config import PROJECT_ROOT
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    ignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "pull_policy: build" in compose, \
        "the image name looks like a Hub repo; without this compose tries to " \
        "pull it, fails, and creates no container at all"
    assert "env_file" in compose, "the key must arrive at run time"
    assert "\n.env\n" in ignore and ".streamlit/secrets.toml" in ignore
    assert "data/index/" in ignore, "a host index must never reach a layer"
