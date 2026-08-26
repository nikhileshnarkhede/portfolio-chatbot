"""
runner.py - run a dataset through the graph and score it.

    from eval.runner import run
    report = run(load_config("exp002_chunk_512"))

Three properties this is built around.

**Every turn is logged before it is scored.** Scoring happens over the run log,
not over live state, so a report can be recomputed from `runs/` without
re-calling the model. Change a metric, rescore last week's run, compare.

**A failed turn is data, not a crash.** Rate limits, a broken prompt, a missing
index for one arm of a sweep - all get recorded and the run continues. An eval
that aborts at question 14 of 22 has told you nothing and spent your quota.

**Single-turn cases get a fresh thread; conversations share one.** Otherwise
case 3 would be answered with case 2's history still in context, and every
score after the first would be measuring something other than what it claims.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from portfolio_chatbot.config import PROJECT_ROOT, AppConfig
from portfolio_chatbot.graph import build_graph, run_turn
from portfolio_chatbot.observability.run_logger import RunLogger

from .metrics import generation, grounding, performance, retrieval, safety

DATASET_DIR = PROJECT_ROOT / "eval" / "datasets"
DEFAULT_DATASETS = (
    "golden_qa", "adversarial_urls", "injection",
    "fabrication_bait", "consistency", "followups",
)


def load_dataset(name: str) -> dict:
    path = Path(name)
    if not path.exists():
        path = DATASET_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {name}. Available: "
            f"{', '.join(sorted(p.stem for p in DATASET_DIR.glob('*.json')))}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _cases(dataset: dict) -> list[tuple[str, list[dict]]]:
    """Normalize both dataset shapes into (conversation_id, [turn_specs])."""
    if "conversations" in dataset:
        return [(c["id"], c["turns"]) for c in dataset["conversations"]]
    return [(c["id"], [c]) for c in dataset.get("cases", [])]


@dataclass
class TurnResult:
    case_id: str
    turn_index: int
    question: str
    answer: str
    retrieval: dict
    generation: dict
    safety: dict
    grounding: dict = field(default_factory=dict)
    injection: dict = field(default_factory=dict)
    consistency: float | None = None
    repeats: int = 1
    leaked_urls: list[str] = field(default_factory=list)
    model_used: str = ""
    fell_back: bool = False
    error: str | None = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    experiment: str
    run_fingerprint: str
    index_fingerprint: str
    dataset: str
    n_turns: int
    n_errors: int
    summary: dict[str, Any]
    turns: list[dict]
    run_dir: str
    elapsed_s: float

    def as_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        s = self.summary
        lines = [
            f"experiment    {self.experiment}",
            f"dataset       {self.dataset}  ({self.n_turns} turns, {self.n_errors} errors)",
            f"run           {self.run_fingerprint}   index {self.index_fingerprint}",
            f"output        {self.run_dir}",
            "",
            "RETRIEVAL",
            f"  route accuracy      {_fmt(s.get('route_correct'))}",
            f"  hit@k               {_fmt(s.get('hit_at_k'))}",
            f"  recall@k            {_fmt(s.get('recall_at_k'))}",
            f"  MRR                 {_fmt(s.get('mrr'))}",
            f"  nDCG@k              {_fmt(s.get('ndcg_at_k'))}",
            f"  type precision      {_fmt(s.get('type_precision'))}",
            f"  context truncated   {s.get('context_truncated_count', 0)} turn(s)",
            f"  modes               {s.get('modes', {})}",
            f"  chunk coverage      {_fmt(s.get('chunk_coverage'))}"
            + (f"   ({s.get('n_reached', 0)}/{s.get('n_indexed', 0)} indexed chunks reached)"
               if s.get("n_indexed") else ""),
            "",
            "GROUNDING  (weight 10)",
            f"  entity grounding    {_fmt(s.get('entity_grounding'))}",
            f"  fabricated rate     {_fmt(s.get('fabricated_entity_rate'))}   <- must be 0",
            f"  ungrounded seen     {(s.get('ungrounded_entities') or [])[:8] or 'none'}",
            f"  faithfulness        {_fmt(s.get('faithfulness'))}",
            "",
            "GENERATION",
            f"  fact coverage       {_fmt(s.get('fact_coverage'))}",
            f"  context overlap     {_fmt(s.get('context_overlap'))}",
            f"  first person        {_fmt(s.get('first_person'))}",
            f"  mean answer chars   {s.get('answer_chars_mean', 0):.0f}",
            f"  relevancy           {_fmt(s.get('relevancy'))}",
            "",
            "SAFETY",
            f"  injection success   {_fmt(s.get('injection_success'))}   <- must be 0",
            f"  persona breaks      {_fmt(s.get('persona_break_rate'))}",
            f"  prompt leak spans   {s.get('prompt_leak_count', 0)}   <- must be 0",
            f"  consistency         {_fmt(s.get('consistency'))}",
            f"  url attempt rate    {_fmt(s.get('attempted_forged_url'))}   <- prompt quality",
            f"  forged urls total   {s.get('forged_url_count_total', 0)}",
            f"  refusal accuracy    {_fmt(s.get('refusal_correct'))}",
            f"  LEAKED urls         {s.get('leaked_url_count', 0)}   <- must be 0",
            "",
            "PERFORMANCE",
            performance.render(performance.PerformanceScores(**s["performance"]))
            if s.get("performance") else "  (not collected)",
            "",
            f"wall clock  {self.elapsed_s:.1f}s",
        ]
        if s.get("leaked_url_count"):
            lines += ["", "*** THE URL GUARD FAILED. Investigate before trusting anything above. ***"]
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "  n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "  n/a" if f != f else f"{f:.3f}"


def _mean(values: list[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float)) and v == v]
    return statistics.fmean(clean) if clean else float("nan")


def run(cfg: AppConfig, dataset_name: str = "golden_qa", *,
        judge: generation.Judge | None = None,
        app: Any | None = None,
        progress: Callable[[str], None] | None = None,
        limit: int | None = None) -> Report:
    """Run `dataset_name` through the pipeline configured by `cfg`."""
    dataset = load_dataset(dataset_name)
    conversations = _cases(dataset)
    if limit:
        conversations = conversations[:limit]

    app = app or build_graph(cfg)
    allowlist = set(cfg.safety.url_allowlist.urls)

    # Loaded once for verbatim-leak detection. Reads whatever prompt this config
    # actually selects, so the check never goes stale against an edited prompt.
    try:
        from portfolio_chatbot.prompts.loader import load_prompt
        system_prompt_text = load_prompt(cfg, cfg.prompts.system).text
    except Exception:
        system_prompt_text = ""

    # Loaded once so chunk coverage can be computed against what is actually
    # indexed. Absent (a fresh index, or a test store) simply omits coverage.
    try:
        from portfolio_chatbot.ingestion.build_index import load_manifest
        manifest = load_manifest(cfg)
    except Exception:
        manifest = {}

    started = time.perf_counter()
    results: list[TurnResult] = []
    records: list[dict] = []
    errors = 0

    with RunLogger.open(cfg, label=dataset_name) as logger:
        for case_id, turns in conversations:
            thread_id = f"{dataset_name}:{case_id}"

            for index, spec in enumerate(turns):
                question = spec["question"]
                repeats = max(1, int(spec.get("repeats", 1)))
                if progress:
                    suffix = f" x{repeats}" if repeats > 1 else ""
                    progress(f"{case_id}[{index}]{suffix} {question[:56]}")

                repeat_answers: list[str] = []
                last_record: dict | None = None
                failure: str | None = None

                for attempt in range(repeats):
                    # A repeated case gets a FRESH thread per run. Sharing one
                    # would let run 2 see run 1's answer in history, which is
                    # exactly the agreement the metric is trying to measure.
                    tid = thread_id if repeats == 1 else f"{thread_id}#{attempt}"
                    try:
                        state = run_turn(cfg, question, thread_id=tid, app=app)
                    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
                        failure = f"{type(exc).__name__}: {exc}"[:400]
                        break
                    last_record = logger.log(
                        state, case_id=case_id, turn_index=index, repeat=attempt
                    )
                    records.append(last_record)
                    repeat_answers.append(last_record.get("answer", ""))

                if last_record is None:
                    errors += 1
                    results.append(TurnResult(
                        case_id=case_id, turn_index=index, question=question,
                        answer="", retrieval={}, generation={}, safety={},
                        error=failure or "no result", repeats=repeats,
                    ))
                    continue

                record = last_record
                r = retrieval.score(record, spec, k=cfg.retrieval.filtered.k)
                g = generation.score(record, spec, judge=judge)
                sf = safety.score(record, spec)
                gr = grounding.score(record, spec)
                inj = safety.injection_success(record, spec, system_prompt_text)
                leaked = safety.leaked_urls(record, allowlist)

                if record.get("error"):
                    errors += 1

                results.append(TurnResult(
                    case_id=case_id, turn_index=index, question=question,
                    answer=record.get("answer", ""),
                    retrieval=r.as_dict(), generation=g.as_dict(), safety=sf.as_dict(),
                    grounding=gr.as_dict(), injection=inj,
                    consistency=grounding.consistency(repeat_answers) if repeats > 1 else None,
                    repeats=repeats,
                    leaked_urls=leaked,
                    model_used=record.get("model_used", ""),
                    fell_back=len(record.get("model_attempts") or []) > 1,
                    error=record.get("error"),
                    latency_s=record.get("latency_s", 0.0) or 0.0,
                    usage=record.get("token_usage") or {},
                ))

        summary = summarize(results, records, manifest)
        run_dir = logger.close(dataset=dataset_name, n_errors=errors, summary=summary)

    report = Report(
        experiment=cfg.run.experiment_name,
        run_fingerprint=cfg.run_fingerprint,
        index_fingerprint=cfg.index_fingerprint,
        dataset=dataset_name,
        n_turns=len(results),
        n_errors=errors,
        summary=summary,
        turns=[r.as_dict() for r in results],
        run_dir=cfg.display_path(run_dir),
        elapsed_s=round(time.perf_counter() - started, 2),
    )
    _write_report(cfg, report)
    return report


def summarize(results: list[TurnResult], records: list[dict] | None = None,
              manifest: dict | None = None) -> dict[str, Any]:
    """Aggregate per-turn scores into the headline numbers.

    D1-D4 aggregate over scored `results`; D5 is delegated to
    `metrics.performance`, computed from the raw run-log `records` so the same
    numbers can be recomputed later from `runs/` without re-calling the model.
    """
    ok = [r for r in results if not r.error or r.retrieval]

    from portfolio_chatbot.observability import perf as perf_registry
    perf = performance.score(records or [], cold_start=perf_registry.snapshot())

    coverage: dict = {}
    if manifest:
        seen = {
            c.get("chunk_id")
            for rec in (records or [])
            for c in (rec.get("chunks") or [])
            if c.get("chunk_id")
        }
        coverage = retrieval.chunk_coverage(seen, manifest)

    def pull(section: str, key: str) -> list[float]:
        return [getattr(r, section).get(key, float("nan")) for r in ok if getattr(r, section)]

    modes: dict[str, int] = {}
    for r in ok:
        mode = r.retrieval.get("mode") or "?"
        modes[mode] = modes.get(mode, 0) + 1

    return {
        "route_correct": _mean(pull("retrieval", "route_correct")),
        "hit_at_k": _mean(pull("retrieval", "hit_at_k")),
        "recall_at_k": _mean(pull("retrieval", "recall_at_k")),
        "mrr": _mean(pull("retrieval", "mrr")),
        "ndcg_at_k": _mean(pull("retrieval", "ndcg_at_k")),
        "type_precision": _mean(pull("retrieval", "type_precision")),
        "context_truncated_count": sum(1 for r in ok if r.retrieval.get("context_truncated")),
        "modes": modes,
        **coverage,

        "fact_coverage": _mean(pull("generation", "fact_coverage")),
        "context_overlap": _mean(pull("generation", "context_overlap")),
        "first_person": _mean(pull("generation", "first_person")),
        "answer_chars_mean": _mean(pull("generation", "answer_chars")),
        "faithfulness": _mean(pull("generation", "faithfulness")),
        "relevancy": _mean(pull("generation", "relevancy")),

        "entity_grounding": _mean(pull("grounding", "entity_grounding")),
        "fabricated_entity_rate": _mean([
            1.0 if r.grounding.get("fabricated") else 0.0 for r in ok if r.grounding
        ]),
        "ungrounded_entities": sorted({
            e for r in ok for e in (r.grounding.get("ungrounded") or [])
        })[:25],
        "injection_success": _mean([
            float(r.injection.get("injection_success", 0.0)) for r in ok if r.injection
        ]),
        "persona_break_rate": _mean([
            1.0 if r.injection.get("persona_break") else 0.0 for r in ok if r.injection
        ]),
        "prompt_leak_count": sum(len(r.injection.get("prompt_leak") or []) for r in ok),
        "consistency": _mean([r.consistency for r in ok if r.consistency is not None]),

        "attempted_forged_url": _mean(pull("safety", "attempted_forged_url")),
        "forged_url_count_total": sum(int(r.safety.get("forged_url_count", 0)) for r in ok),
        "refusal_correct": _mean(pull("safety", "refusal_correct")),
        "leaked_url_count": sum(len(r.leaked_urls) for r in results),

        "fell_back_count": sum(1 for r in results if r.fell_back),
        "n_errors": sum(1 for r in results if r.error),

        # ---- D5, from metrics/performance.py ----
        "performance": perf.as_dict(),
        "ttft_p50": perf.ttft.get("p50"),
        "ttft_p95": perf.ttft.get("p95"),
        "latency_p50": perf.latency.get("p50"),
        "latency_p95": perf.latency.get("p95"),
        "latency_mean": perf.latency.get("mean"),
        "input_tokens_total": perf.input_tokens_total,
        "output_tokens_total": perf.output_tokens_total,
        "tokens_per_turn_mean": perf.tokens_per_turn_mean,
        "cost_usd_total": perf.cost_usd_total,
        "usage_reported": perf.usage_reported,
        "fallback_rate": perf.fallback_rate,
        "error_rate": perf.error_rate,
    }


def _write_report(cfg: AppConfig, report: Report) -> Path:
    directory = cfg.resolve(cfg.paths.eval_reports_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.run.experiment_name}__{report.dataset}__{cfg.run_fingerprint}"
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    (directory / f"{stem}.txt").write_text(report.render(), encoding="utf-8")
    return path


__all__ = ["run", "Report", "TurnResult", "load_dataset", "summarize", "DEFAULT_DATASETS"]
