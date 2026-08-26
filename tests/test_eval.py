"""
Tests for the evaluation layer.

A metric that is quietly wrong is worse than no metric: it produces confident
numbers that send you optimizing in the wrong direction. So the scorers are
tested against hand-built inputs with known answers, and the full runner is
exercised end to end offline.

`test_chunk_refs_must_carry_identity` is the regression guard for a bug this
suite was built to find. `chunk_refs` originally omitted `identity`, the field
every retrieval metric joins on, so the first real eval reported hit@k = 0.000
across all 22 cases. The pipeline was fine; the instrument was broken - which is
the single most dangerous failure mode an eval layer has.
"""

from __future__ import annotations


import pytest
from langchain_core.messages import AIMessage

from eval import compare as compare_mod
from eval.metrics import generation, retrieval, safety
from eval.runner import load_dataset, run
from portfolio_chatbot.config import load_config
from portfolio_chatbot.state import chunk_refs

ALLOWED = "https://github.com/nikhileshnarkhede"
FORGED = "https://github.com/fake/not-real"


def chunks(*identities, chunk_type="project"):
    return [{"rank": i, "identity": ident, "chunk_type": chunk_type, "chunk_id": f"c{i}"}
            for i, ident in enumerate(identities)]


# ---------------------------------------------------------------- retrieval

def test_hit_at_k_matches_identity_substring():
    """Golden cases name a project; the chunk identity is a join of attributes."""
    found = chunks("Chat with YouTube Videos | RAG / Conversational AI")
    assert retrieval.hit_at_k(found, ["Chat with YouTube Videos"]) == 1.0


def test_hit_at_k_is_case_insensitive():
    assert retrieval.hit_at_k(chunks("SymbOptAI | Symbolic Regression"), ["symboptai"]) == 1.0


def test_hit_at_k_zero_when_absent():
    assert retrieval.hit_at_k(chunks("Supply Chain Tracker"), ["SymbOptAI"]) == 0.0


def test_recall_distinguishes_partial_from_complete():
    """hit@k cannot tell 1-of-11 from 11-of-11; recall is why it exists."""
    found = chunks("A", "B")
    assert retrieval.hit_at_k(found, ["A", "C"]) == 1.0
    assert retrieval.recall_at_k(found, ["A", "C"]) == 0.5


def test_mrr_rewards_rank():
    first = chunks("target", "other")
    third = chunks("other", "other2", "target")
    assert retrieval.mrr(first, ["target"]) == 1.0
    assert retrieval.mrr(third, ["target"]) == pytest.approx(1 / 3)


def test_type_precision_penalises_off_type_context():
    mixed = chunks("A") + chunks("B", chunk_type="experience")
    assert retrieval.type_precision(mixed, ["project"]) == 0.5


def test_route_correct_handles_expected_no_match():
    """expected_route=None asserts the question SHOULD reach MMR."""
    assert retrieval.route_correct({"rule_name": None}, None) == 1.0
    assert retrieval.route_correct({"rule_name": "projects"}, None) == 0.0


def test_score_reports_which_chunks_were_missing():
    turn = {"chunks": chunks("A"), "route": {"rule_name": "projects"}}
    case = {"expected_identities": ["A", "B"], "expected_route": "projects"}
    assert retrieval.score(turn, case).missing == ["B"]


# ---------------------------------------------------------------- safety

def test_attempted_forged_url_reads_the_audit_not_the_answer():
    """The metric measures what the model TRIED, which only the audit records."""
    turn = {"answer": "clean text", "link_audit": {"stripped": [FORGED], "kept": []}}
    scored = safety.score(turn, {})
    assert scored.attempted_forged_url == 1.0
    assert scored.forged_url_count == 1


def test_no_attempt_scores_zero():
    turn = {"answer": "fine", "link_audit": {"stripped": [], "kept": [ALLOWED]}}
    assert safety.score(turn, {}).attempted_forged_url == 0.0


def test_leaked_urls_is_empty_when_the_guard_works():
    turn = {"answer": f"see {ALLOWED}"}
    assert safety.leaked_urls(turn, {ALLOWED}) == []


def test_leaked_urls_catches_a_broken_guard():
    """If this ever fires in a real run, the guard failed - not the model."""
    turn = {"answer": f"see {FORGED}"}
    assert safety.leaked_urls(turn, {ALLOWED}) == [FORGED]


def test_refusal_detection():
    assert safety.looks_like_refusal("I don't have that detail here right now")
    assert not safety.looks_like_refusal("I built eleven projects.")


def test_refusal_accuracy_scores_over_refusal_too():
    """Refusing an answerable question is a failure, not a safe default."""
    answerable = {"answer": "I don't have that", "link_audit": {}}
    assert safety.score(answerable, {"expect_refusal": False}).refusal_correct == 0.0


# ---------------------------------------------------------------- generation

def test_fact_coverage_and_missing_facts():
    turn = {"answer": "I worked at Choir Corp on ML."}
    scored = generation.score(turn, {"must_contain": ["Choir", "Bajaj"]})
    assert scored.fact_coverage == 0.5
    assert scored.missing_facts == ["Bajaj"]


def test_first_person_flags_third_person_drift():
    """The persona prompt insists the bot IS Nikhilesh."""
    assert generation.first_person("I built several projects.") == 1.0
    assert generation.first_person("Nikhilesh has experience in ML.") == 0.0


def test_judge_is_optional():
    scored = generation.score({"answer": "hi", "context": "hi"}, {})
    assert scored.faithfulness != scored.faithfulness  # NaN


def test_judge_failure_does_not_sink_tier_one_scores():
    class Broken:
        def faithfulness(self, *a): raise RuntimeError("judge down")
        def relevancy(self, *a): raise RuntimeError("judge down")

    scored = generation.score({"answer": "I built things", "context": "things"},
                              {"must_contain": ["built"]}, judge=Broken())
    assert scored.fact_coverage == 1.0


# ---------------------------------------------------------------- datasets

@pytest.mark.parametrize("name", ["golden_qa", "adversarial_urls", "followups"])
def test_datasets_parse(name):
    assert load_dataset(name)


def test_golden_cases_have_unique_ids():
    ids = [c["id"] for c in load_dataset("golden_qa")["cases"]]
    assert len(set(ids)) == len(ids)


def test_expected_routes_exist_in_the_config(cfg):
    """A typo'd expected_route would score 0 forever and look like a real failure."""
    known = {r.name for r in cfg.retrieval.routing.routes} | {None}
    for case in load_dataset("golden_qa")["cases"]:
        assert case.get("expected_route") in known, case["id"]


def test_expected_chunk_types_exist_in_the_index(cfg):
    valid = set(cfg.ingestion.structural.type_map.values()) | {
        f"{s}_intro" for s in cfg.ingestion.structural.section_tags
    } | set(cfg.ingestion.structural.section_tags)
    for case in load_dataset("golden_qa")["cases"]:
        for t in case.get("expected_chunk_types") or []:
            assert t in valid, f"{case['id']}: unknown chunk_type {t!r}"


# ---------------------------------------------------------------- state join

def test_chunk_refs_must_carry_identity():
    """REGRESSION: omitting `identity` made every retrieval metric score 0.000."""
    from langchain_core.documents import Document
    state = {"documents": [Document(page_content="x",
                                    metadata={"identity": "SymbOptAI", "chunk_id": "abc"})]}
    refs = chunk_refs(state)
    assert refs[0]["identity"] == "SymbOptAI"
    assert refs[0]["chunk_id"] == "abc"
    assert retrieval.hit_at_k(refs, ["SymbOptAI"]) == 1.0


# ---------------------------------------------------------------- runner

@pytest.fixture
def eval_cfg(tmp_path, cfg):
    """Config writing runs and reports into a temp directory."""
    return load_config(overrides=[
        f'paths.runs_dir="{tmp_path.as_posix()}/runs"',
        f'paths.eval_reports_dir="{tmp_path.as_posix()}/reports"',
    ])


@pytest.fixture
def offline(monkeypatch, fake_index):
    class LLM:
        def invoke(self, _p):
            return AIMessage(content="earlier turns")

        def stream(self, _p):
            for w in ["I built ", "projects. ", f"See [x]({FORGED})."]:
                yield AIMessage(content=w)

    for module in ("generate", "summarize"):
        monkeypatch.setattr(f"portfolio_chatbot.nodes.{module}.build_llm",
                            lambda c, model=None: LLM())


def test_runner_scores_a_full_dataset(eval_cfg, offline, fake_index):
    report = run(eval_cfg, "golden_qa", limit=4)
    assert report.n_turns == 4
    assert report.n_errors == 0
    assert 0.0 <= report.summary["route_correct"] <= 1.0
    assert report.summary["leaked_url_count"] == 0


def test_runner_detects_the_forged_url_attempts(eval_cfg, offline, fake_index):
    """The fake model always emits a forged link, so the rate must be 1.0."""
    report = run(eval_cfg, "golden_qa", limit=3)
    assert report.summary["attempted_forged_url"] == 1.0
    assert report.summary["leaked_url_count"] == 0, "guard must still hold"


def test_runner_writes_a_replayable_log(eval_cfg, offline, fake_index):
    from portfolio_chatbot.observability.run_logger import load_turns

    report = run(eval_cfg, "golden_qa", limit=3)
    turns = load_turns(eval_cfg.project_root / report.run_dir)
    assert len(turns) == 3
    assert all("chunks" in t and "draft_answer" in t for t in turns)


def test_runner_survives_a_failing_turn(eval_cfg, offline, fake_index, monkeypatch):
    """One bad question must not abort a 22-question run."""
    import eval.runner as runner_mod

    calls = {"n": 0}
    real = runner_mod.run_turn

    def flaky(cfg, question, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return real(cfg, question, **kw)

    monkeypatch.setattr(runner_mod, "run_turn", flaky)
    report = run(eval_cfg, "golden_qa", limit=3)
    assert report.n_turns == 3
    assert report.n_errors == 1


def test_conversations_share_a_thread(eval_cfg, offline, fake_index):
    """followups.json depends on history carrying between its turns."""
    report = run(eval_cfg, "followups", limit=1)
    assert report.n_turns == 2
    assert {t["case_id"] for t in report.turns} == {"followup_project_detail"}


# ---------------------------------------------------------------- compare

def _report(name, **summary):
    base = {"route_correct": 1.0, "hit_at_k": 1.0, "attempted_forged_url": 0.0}
    base.update(summary)
    return {
        "experiment": name, "dataset": "golden_qa", "summary": base,
        "turns": [{"case_id": "a", "turn_index": 0, "question": "q",
                   "retrieval": {"hit_at_k": base["hit_at_k"]}}],
    }


def test_compare_labels_improvement_and_regression():
    result = compare_mod.compare(_report("base", hit_at_k=0.5), _report("cand", hit_at_k=0.9))
    row = next(r for r in result["metrics"] if r["metric"] == "hit_at_k")
    assert row["verdict"] == "better" and row["delta"] == pytest.approx(0.4)


def test_compare_knows_lower_is_better_for_url_attempts():
    result = compare_mod.compare(_report("base", attempted_forged_url=0.4),
                                 _report("cand", attempted_forged_url=0.1))
    row = next(r for r in result["metrics"] if r["metric"] == "attempted_forged_url")
    assert row["verdict"] == "better"


def test_compare_surfaces_per_case_regressions():
    """An improved average can hide the case you actually care about."""
    result = compare_mod.compare(_report("base", hit_at_k=1.0), _report("cand", hit_at_k=0.0))
    assert any(c["verdict"] == "WORSE" for c in result["case_changes"])


# ---------------------------------------------------------------- repeats (stage 3)

def test_repeated_cases_use_independent_threads(eval_cfg, offline, fake_index):
    """A shared thread would let run 2 see run 1's answer in history - which is
    precisely the agreement the consistency metric is trying to measure."""
    report = run(eval_cfg, "consistency", limit=2)
    assert report.n_turns == 2
    for t in report.turns:
        assert t["repeats"] == 3
        assert t["consistency"] is not None


def test_repeats_log_every_run_not_just_the_last(eval_cfg, offline, fake_index):
    from portfolio_chatbot.observability.run_logger import load_turns
    report = run(eval_cfg, "consistency", limit=2)
    turns = load_turns(eval_cfg.project_root / report.run_dir)
    assert len(turns) == 6, "2 cases x 3 repeats must all reach the run log"


def test_injection_suite_scores_injection_success(eval_cfg, offline, fake_index):
    report = run(eval_cfg, "injection", limit=3)
    assert report.summary["injection_success"] == 0.0
    assert report.summary["prompt_leak_count"] == 0


def test_grounding_is_scored_on_every_suite(eval_cfg, offline, fake_index):
    report = run(eval_cfg, "fabrication_bait", limit=3)
    assert "entity_grounding" in report.summary
    assert 0.0 <= report.summary["fabricated_entity_rate"] <= 1.0


# ---------------------------------------------------------------- nDCG / coverage (stage 6)

def _graded(*identities):
    return [{"rank": i, "identity": ident, "chunk_id": f"c{i}", "chunk_type": "project"}
            for i, ident in enumerate(identities)]


def test_ndcg_rewards_putting_the_primary_chunk_first():
    """MRR and hit@k cannot see the difference; nDCG is why graded relevance exists."""
    case = {"expected_identities": ["A", "B"], "primary_identities": ["A"]}
    first = retrieval.ndcg_at_k(_graded("A", "B", "X"), case)
    last = retrieval.ndcg_at_k(_graded("X", "B", "A"), case)
    assert first == 1.0
    assert last < first


def test_ndcg_is_zero_when_nothing_relevant_is_found():
    case = {"expected_identities": ["A"], "primary_identities": ["A"]}
    assert retrieval.ndcg_at_k(_graded("X", "Y"), case) == 0.0


def test_ndcg_degrades_to_binary_without_grades():
    """A case that declares no primaries must still produce a usable number."""
    assert retrieval.ndcg_at_k(_graded("A"), {"expected_identities": ["A"]}) == 1.0


def test_ndcg_is_nan_without_expectations():
    scored = retrieval.ndcg_at_k(_graded("A"), {})
    assert scored != scored


def test_chunk_coverage_finds_orphans():
    """Content no query reaches is invisible to every question-driven metric."""
    manifest = {"chunks": [{"chunk_id": c, "identity": c.upper()} for c in ("a", "b", "c")]}
    cov = retrieval.chunk_coverage({"a", "b"}, manifest)
    assert cov["chunk_coverage"] == pytest.approx(2 / 3)
    assert [o["identity"] for o in cov["orphans"]] == ["C"]


def test_chunk_coverage_without_a_manifest_is_nan():
    cov = retrieval.chunk_coverage(set(), {})
    assert cov["chunk_coverage"] != cov["chunk_coverage"]


def test_golden_set_was_extended():
    cases = load_dataset("golden_qa")["cases"]
    assert len(cases) >= 40
    assert any(c.get("primary_identities") for c in cases)


def test_primary_identities_are_a_subset_of_expected():
    """A primary chunk that is not also expected would be graded but never sought."""
    for case in load_dataset("golden_qa")["cases"]:
        expected = set(case.get("expected_identities") or [])
        for primary in case.get("primary_identities") or []:
            assert primary in expected, f"{case['id']}: {primary!r} not in expected_identities"
