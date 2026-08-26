"""
panels.py - the six dashboard tabs.

Presentation only. Every decision - whether a metric passes, whether a delta is
significant, whether an apply is allowed - is made in `eval/` and rendered here.
That separation is what lets the same rules run headless in CI.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from eval import apply as apply_mod
from eval import calibration as calib_mod
from eval import gate as gate_mod
from eval import noise as noise_mod
from eval import ratings as ratings_mod
from eval.runner import load_dataset, run
from portfolio_chatbot.config import AppConfig

from . import state, tunables

PASS, WARN, FAIL, NEUTRAL = "pass", "warn", "fail", "neutral"


def inject_styles() -> None:
    st.markdown("""
<style>
  .metric-card { border:1px solid rgba(128,128,128,.25); border-radius:4px;
                 padding:.55rem .7rem; border-top:3px solid var(--c); }
  .metric-card .lbl { font-size:.66rem; letter-spacing:.06em; text-transform:uppercase;
                      opacity:.65; }
  .metric-card .val { font-size:1.35rem; font-weight:600; font-variant-numeric:tabular-nums;
                      color:var(--c); line-height:1.3; }
  .metric-card .sub { font-size:.68rem; opacity:.6; font-variant-numeric:tabular-nums; }
  .pass    { --c:#17a06a; }
  .warn    { --c:#c99012; }
  .fail    { --c:#d0453a; }
  .neutral { --c:#7a8794; }
  .knob-changed { color:#c99012; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# helpers
# ==========================================================================

def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _verdict(metric: str, value: Any, spec: dict) -> tuple[str, str]:
    """(state, requirement) for one metric, from thresholds.json."""
    v = _num(value)
    if v is None:
        return NEUTRAL, ""

    inv = (spec.get("invariants") or {}).get(metric)
    if inv:
        if "max" in inv and v > inv["max"]:
            return FAIL, f"must be <= {inv['max']}"
        if "min" in inv and v < inv["min"]:
            return FAIL, f"must be >= {inv['min']}"
        return PASS, "invariant"

    rule = (spec.get("thresholds") or {}).get(metric)
    if not rule:
        return NEUTRAL, ""
    bad = ("min" in rule and v < rule["min"]) or ("max" in rule and v > rule["max"])
    requirement = f">= {rule['min']}" if "min" in rule else f"<= {rule['max']}"
    if not bad:
        return PASS, requirement
    return (FAIL if rule.get("on_fail") == "block" else WARN), requirement


def metric_card(label: str, value: Any, spec: dict, metric: str,
                sub: str = "") -> None:
    tone, requirement = _verdict(metric, value, spec)
    v = _num(value)
    shown = "n/a" if v is None else (f"{v:.0f}" if abs(v) >= 100 else f"{v:.3f}")
    st.markdown(
        f'<div class="metric-card {tone}"><div class="lbl">{label}</div>'
        f'<div class="val">{shown}</div>'
        f'<div class="sub">{sub or requirement}</div></div>',
        unsafe_allow_html=True)


def _grid(items: list[tuple[str, str]], summary: dict, spec: dict,
          subs: dict[str, str] | None = None, per_row: int = 4) -> None:
    subs = subs or {}
    for i in range(0, len(items), per_row):
        for col, (metric, label) in zip(st.columns(per_row), items[i:i + per_row]):
            with col:
                metric_card(label, summary.get(metric), spec, metric, subs.get(metric, ""))


# ==========================================================================
# sidebar
# ==========================================================================

def sidebar(cfg: AppConfig | None, working: dict, base: dict, error: str | None) -> None:
    with st.sidebar:
        st.subheader("Configuration")

        changes = tunables.changed_knobs(working, base)
        if not changes:
            st.caption("Matches the live config.")
        else:
            st.warning(f"{len(changes)} unsaved change(s)")
            for knob, was, now in changes:
                st.markdown(f"- **{knob.label}**: `{was}` → `{now}`")

        if cfg:
            st.divider()
            st.caption(f"run `{cfg.run_fingerprint}`")
            st.caption(f"index `{cfg.index_fingerprint}`")
            if cfg.index_path.exists():
                st.success("Index ready")
            else:
                st.error("No index for this config")
                st.code("python scripts/ingest.py", language="bash")

        st.divider()
        calib = calib_mod.load()
        if calib and _num(calib.kappa) is not None:
            label = {"trust": "✔ trusted", "direction": "~ direction only",
                     "suppress": "✕ suppressed"}.get(calib.verdict, calib.verdict)
            st.caption(f"Judge κ = {calib.kappa:.2f} · {label}")
        else:
            st.caption("Judge not calibrated — judged metrics cannot block.")

        st.divider()
        if st.button("Reset to live config", use_container_width=True):
            state.reset_to_default()
            state.clear_results()
            st.rerun()
        if error:
            st.error("Config invalid")


# ==========================================================================
# 1 · Tune
# ==========================================================================

def tune(working: dict, base: dict, cfg: AppConfig | None) -> None:
    st.caption("Adjust parameters. Nothing is written until you Apply.")

    changed = {k.path for k, _, _ in tunables.changed_knobs(working, base)}
    updated = working

    for group in tunables.GROUPS:
        st.subheader(group)
        knobs = [k for k in tunables.KNOBS if k.group == group]
        for row in range(0, len(knobs), 2):
            for col, knob in zip(st.columns(2), knobs[row:row + 2]):
                with col:
                    updated = _render_knob(updated, knob, knob.path in changed)
        st.divider()

    if updated != working:
        state.update(updated)
        state.clear_results()
        st.rerun()

    if tunables.requires_reingest(working, base):
        st.warning(
            "A chunking parameter changed, so this config needs its own vector "
            "index. Results scored against the existing index would be measuring "
            "the old chunking.",
            icon="⚠",
        )
        name = cfg.run.experiment_name if cfg else "default"
        st.code(f"python scripts/ingest.py --experiment {name}", language="bash")


def _render_knob(config: dict, knob: tunables.Knob, is_changed: bool) -> dict:
    current = tunables.get_value(config, knob.path)
    label = f"{knob.label} ●" if is_changed else knob.label
    key = f"knob_{knob.path}"

    if knob.kind == "toggle":
        value = st.toggle(label, value=bool(current), help=knob.help, key=key)
    elif knob.kind == "select":
        options = list(knob.options or ())
        index = options.index(current) if current in options else 0
        value = st.selectbox(label, options, index=index, help=knob.help, key=key)
    else:
        is_float = isinstance(current, float) or (knob.step and knob.step < 1)
        value = st.slider(
            label,
            min_value=float(knob.minimum) if is_float else int(knob.minimum),
            max_value=float(knob.maximum) if is_float else int(knob.maximum),
            value=float(current) if is_float else int(current),
            step=float(knob.step) if is_float else int(knob.step),
            help=knob.help, key=key,
        )

    if knob.rebuilds_index:
        st.caption("↻ rebuilds the index")
    return tunables.set_value(config, knob.path, value) if value != current else config


# ==========================================================================
# 2 · Run
# ==========================================================================

def run_tab(cfg: AppConfig | None, datasets: tuple[str, ...]) -> None:
    if cfg is None:
        st.info("Fix the configuration before running.")
        return

    chosen = st.multiselect("Suites", list(datasets), default=["golden_qa"])
    left, right = st.columns(2)
    repeats = left.number_input(
        "Repeats", 1, 10, 1,
        help="More than one measures the noise floor: sigma per metric, and the "
             "minimum detectable effect. The spec calls for 5 before any "
             "comparison is believed.")
    use_judge = right.toggle(
        "Use the LLM judge", value=False,
        help="Adds faithfulness and relevancy at roughly 2 extra calls per turn, "
             "on the judge model's own quota.")

    n_cases = sum(len(load_dataset(d).get("cases", [])) or
                  sum(len(c["turns"]) for c in load_dataset(d).get("conversations", []))
                  for d in chosen) if chosen else 0
    generation_calls = n_cases * repeats
    judge_calls = n_cases * repeats * 2 if use_judge else 0

    st.caption(
        f"Projected: **{generation_calls}** generation calls on `{cfg.llm.primary_model}`"
        + (f", **{judge_calls}** judge calls on `{cfg.judge_model}`" if judge_calls else "")
        + ". Groq meters each model separately, so these draw on different quotas."
    )
    if generation_calls > 900:
        st.warning("This exceeds a typical daily free-tier quota for one model.", icon="⚠")

    if not cfg.index_path.exists():
        st.error("No index for this config. Ingest before running.")
        return

    if st.button("Run", type="primary", disabled=not chosen):
        judge = None
        if use_judge:
            from eval.metrics.judge import build_judge
            judge = build_judge(cfg)

        progress = st.progress(0.0, text="starting")
        try:
            if repeats > 1:
                floor = noise_mod.measure(
                    lambda i: run(cfg, chosen[0], judge=judge),
                    repeats=int(repeats), dataset=chosen[0],
                    progress=lambda line: progress.progress(0.5, text=line))
                st.session_state.last_noise = floor
            report = run(cfg, chosen[0], judge=judge,
                         progress=lambda line: progress.progress(0.9, text=line))
        except Exception as exc:
            progress.empty()
            st.error(f"Run failed: {exc}")
            return

        progress.empty()
        st.session_state.last_report = report
        st.session_state.last_gate = gate_mod.check(
            report, gate_mod.load_baseline(chosen[0]), kappa=calib_mod.current_kappa())
        st.success(f"{report.n_turns} turns scored. See the Scorecard tab.")

    floor = st.session_state.get("last_noise")
    if floor:
        with st.expander("Noise floor", expanded=False):
            st.code(floor.render(), language=None)


# ==========================================================================
# 3 · Scorecard
# ==========================================================================

DIMENSIONS = (
    ("Grounding · weight 10", [
        ("entity_grounding", "entity grounding"), ("fabricated_entity_rate", "fabricated rate"),
        ("faithfulness", "faithfulness"), ("context_overlap", "context overlap"),
    ]),
    ("Safety · weight 10", [
        ("injection_success", "injection success"), ("leaked_url_count", "leaked urls"),
        ("attempted_forged_url", "url attempts"), ("refusal_correct", "refusal accuracy"),
        ("consistency", "consistency"), ("persona_break_rate", "persona breaks"),
        ("prompt_leak_count", "prompt leaks"),
    ]),
    ("Retrieval · weight 9", [
        ("route_correct", "route accuracy"), ("hit_at_k", "hit@k"),
        ("recall_at_k", "recall@k"), ("ndcg_at_k", "nDCG@k"),
        ("type_precision", "type precision"), ("chunk_coverage", "chunk coverage"),
    ]),
    ("Usefulness · weight 8", [
        ("human_rating_mean", "human mean"), ("human_rating_min", "human min"),
        ("fact_coverage", "fact coverage"), ("first_person", "first person"),
    ]),
    ("Performance · weight 8", [
        ("ttft_p95", "TTFT p95"), ("latency_p95", "latency p95"),
        ("tokens_per_turn_mean", "tokens/turn"), ("error_rate", "error rate"),
    ]),
)


def scorecard(cfg: AppConfig | None) -> None:
    report = st.session_state.get("last_report")
    if not report:
        st.info("Run a suite to populate the scorecard.")
        return

    spec = gate_mod.load_thresholds()
    summary = dict(report.summary)
    if cfg:
        summary.update(ratings_mod.summary_fields(cfg.run_fingerprint))

    result = st.session_state.get("last_gate")
    if result:
        (st.success if result.passed else st.error)(
            result.render().splitlines()[0])

    for title, metrics in DIMENSIONS:
        st.subheader(title)
        _grid(metrics, summary, spec)
        st.write("")

    if result and result.violations:
        with st.expander(f"Violations ({len(result.violations)})", expanded=not result.passed):
            st.code(result.render(), language=None)

    ungrounded = summary.get("ungrounded_entities") or []
    if ungrounded:
        with st.expander(f"Ungrounded entities ({len(ungrounded)})", expanded=True):
            st.caption("Named in an answer but absent from the retrieved context.")
            st.write(", ".join(f"`{e}`" for e in ungrounded))

    with st.expander("Per-case detail"):
        st.dataframe(
            [{"case": t["case_id"], "route": (t["retrieval"] or {}).get("route_correct"),
              "hit@k": (t["retrieval"] or {}).get("hit_at_k"),
              "grounding": (t["grounding"] or {}).get("entity_grounding"),
              "answer": (t.get("answer") or "")[:90]}
             for t in report.turns],
            use_container_width=True, hide_index=True)


# ==========================================================================
# 4 · Compare
# ==========================================================================

def compare(cfg: AppConfig | None) -> None:
    report = st.session_state.get("last_report")
    if not report:
        st.info("Run a suite first.")
        return

    baseline = gate_mod.load_baseline(report.dataset)
    if not baseline:
        st.info("No baseline recorded for this suite yet.")
        if st.button("Save this run as the baseline"):
            gate_mod.save_baseline(
                report,
                sigmas=(st.session_state.get("last_noise").sigmas
                        if st.session_state.get("last_noise") else None))
            st.rerun()
        return

    floor = st.session_state.get("last_noise")
    st.caption(
        "Deltas smaller than the minimum detectable effect are marked "
        "**inconclusive** — that is measurement error, not an improvement."
        if floor else
        "No noise floor measured, so significance cannot be judged. "
        "Run with repeats ≥ 5."
    )

    rows = []
    spec = gate_mod.load_thresholds()
    for metric in (spec.get("thresholds") or {}):
        before, after = _num(baseline["summary"].get(metric)), _num(report.summary.get(metric))
        if before is None or after is None:
            continue
        delta = after - before
        higher_better = "min" in spec["thresholds"][metric]

        if floor and not floor.is_significant(metric, delta):
            verdict = "inconclusive"
        elif abs(delta) < 1e-9:
            verdict = "unchanged"
        else:
            verdict = "better" if (delta > 0) == higher_better else "WORSE"

        rows.append({"metric": metric, "baseline": round(before, 4),
                     "candidate": round(after, 4), "delta": round(delta, 4),
                     "verdict": verdict})

    rows.sort(key=lambda r: (r["verdict"] != "WORSE", -abs(r["delta"])))
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if any(r["verdict"] == "WORSE" for r in rows):
        st.error("Some metrics regressed. Check the per-case table before accepting.")


# ==========================================================================
# 5 · Rate
# ==========================================================================

def rate(cfg: AppConfig | None) -> None:
    report = st.session_state.get("last_report")
    if not report or cfg is None:
        st.info("Run a suite first — you rate its answers here.")
        return

    st.caption(
        "Blind: the config that produced each answer is hidden, so you rate the "
        "text rather than the experiment you hope is winning. Ratings are keyed "
        "by answer hash, so an unchanged answer keeps its score across runs."
    )

    pending = ratings_mod.unrated(report.turns, cfg.run_fingerprint)
    stats = ratings_mod.stats(cfg.run_fingerprint)

    left, mid, right = st.columns(3)
    left.metric("Rated", stats.n)
    mid.metric("Mean", f"{stats.mean:.1f}" if stats.n else "—")
    right.metric("Awaiting rating", len(pending))

    if stats.below_three:
        st.error(f"Blocking: {len(stats.below_three)} answer(s) scored below 3 — "
                 f"{', '.join(stats.below_three[:5])}")

    if not pending:
        st.success("Every answer in this run has been rated.")
        return

    turn = pending[min(st.session_state.rate_index, len(pending) - 1)]
    st.divider()
    st.markdown(f"**Question**  \n{turn['question']}")
    st.markdown("**Answer**")
    st.info(turn.get("answer") or "(empty)")

    with st.expander("Retrieved context"):
        st.text((turn.get("answer") and report.turns and
                 next((r for r in report.turns if r["case_id"] == turn["case_id"]), {})
                 .get("answer", "")) or "")

    score = st.slider("Score", 0, 10, 7, key=f"rate_{turn['case_id']}")
    st.caption(ratings_mod.rubric_for(score))
    note = st.text_input("Note (optional)", key=f"note_{turn['case_id']}")

    a, b = st.columns(2)
    if a.button("Save and next", type="primary", use_container_width=True):
        ratings_mod.save_rating(cfg.run_fingerprint, turn["case_id"],
                                turn.get("answer", ""), score, note,
                                experiment=cfg.run.experiment_name)
        st.rerun()
    if b.button("Skip", use_container_width=True):
        st.session_state.rate_index += 1
        st.rerun()


# ==========================================================================
# 6 · Apply
# ==========================================================================

def apply_tab(cfg: AppConfig | None, working: dict, base: dict) -> None:
    st.caption("You cannot apply a config you have not measured.")

    changes = tunables.changed_knobs(working, base)
    if not changes:
        st.info("The working config matches the live one — nothing to apply.")
    else:
        st.dataframe(
            [{"parameter": k.label, "live": str(was), "candidate": str(now),
              "re-ingest": "yes" if k.rebuilds_index else ""}
             for k, was, now in changes],
            use_container_width=True, hide_index=True)
        if len(changes) > 1:
            st.warning(
                f"{len(changes)} parameters changed at once. If this wins you will "
                f"not know which one did it — the spec asks for one variable per "
                f"experiment.", icon="⚠")

    report = st.session_state.get("last_report")
    result = st.session_state.get("last_gate")
    problems = apply_mod.check_ready(report, result, cfg) if cfg else ["Config is invalid."]

    st.divider()
    if problems:
        st.error("Apply is blocked:")
        for problem in problems:
            st.markdown(f"- {problem}")
    else:
        st.success("All checks passed. This config is ready to go live.")

    label = st.text_input("Name this change", placeholder="e.g. wider retrieval k")
    override = ""
    if problems:
        override = st.text_input(
            "Override note",
            placeholder="Explain why these problems are acceptable",
            help="An override has to be written down, not clicked. It is recorded "
                 "in configs/provenance.json alongside the failures it bypassed.")

    if st.button("Apply to live config", type="primary",
                 disabled=cfg is None or (bool(problems) and not override.strip())):
        try:
            applied = apply_mod.apply(cfg, working, report, result, label, override)
        except apply_mod.ApplyRefused as exc:
            st.error(str(exc))
            return
        st.success("Applied.")
        st.code(applied.render(), language=None)
        state.reset_to_default()

    history = apply_mod.list_history()
    if history:
        with st.expander(f"History and rollback ({len(history)})"):
            choice = st.selectbox("Archived config", [p.name for p in history])
            if st.button("Roll back to this"):
                apply_mod.rollback(next(p for p in history if p.name == choice))
                state.reset_to_default()
                st.success(f"Rolled back to {choice}")
                st.rerun()


# ==========================================================================
# ⚙ Thresholds
# ==========================================================================

def thresholds() -> None:
    st.caption("You own these numbers. Editing one re-scores the current "
               "scorecard immediately — no re-run, since the metrics already exist.")

    spec = gate_mod.load_thresholds()
    edited = json.loads(json.dumps(spec))
    dirty = False

    st.subheader("Invariants")
    st.caption("No tolerance. These block unconditionally, baseline or not.")
    for metric, rule in spec["invariants"].items():
        st.markdown(f"**{metric}** — {rule.get('why', '')}")

    st.divider()
    st.subheader("Thresholds")
    for metric, rule in spec["thresholds"].items():
        cols = st.columns([3, 2, 2])
        cols[0].markdown(f"**{metric}**  \n<small>weight {rule.get('weight', '—')}</small>",
                         unsafe_allow_html=True)
        bound = "min" if "min" in rule else "max"
        value = cols[1].number_input(
            f"{bound}", value=float(rule[bound]), key=f"th_{metric}",
            step=0.01, format="%.3f", label_visibility="collapsed")
        on_fail = cols[2].selectbox(
            "on fail", ["block", "warn"],
            index=0 if rule.get("on_fail") == "block" else 1,
            key=f"of_{metric}", label_visibility="collapsed")
        if value != rule[bound] or on_fail != rule.get("on_fail"):
            edited["thresholds"][metric][bound] = value
            edited["thresholds"][metric]["on_fail"] = on_fail
            dirty = True

    if dirty and st.button("Save thresholds", type="primary"):
        edited["_meta"]["status"] = "edited in the dashboard"
        gate_mod.THRESHOLDS_PATH.write_text(
            json.dumps(edited, indent=2), encoding="utf-8")
        report = st.session_state.get("last_report")
        if report:
            st.session_state.last_gate = gate_mod.check(
                report, gate_mod.load_baseline(report.dataset),
                kappa=calib_mod.current_kappa())
        st.success("Saved and re-scored.")
        st.rerun()


__all__ = ["inject_styles", "sidebar", "tune", "run_tab", "scorecard",
           "compare", "rate", "apply_tab", "thresholds"]
