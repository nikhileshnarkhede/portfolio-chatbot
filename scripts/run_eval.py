"""
Run an evaluation dataset, or compare two runs.

    python scripts/run_eval.py                                   # baseline, golden_qa
    python scripts/run_eval.py --experiment exp002_chunk_512
    python scripts/run_eval.py --dataset adversarial_urls
    python scripts/run_eval.py --all                             # every dataset
    python scripts/run_eval.py --compare exp001_baseline exp002_chunk_512

This calls the real model, so it spends Groq quota: roughly one call per turn,
22 turns for golden_qa. Use --limit while iterating on a dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval import compare as compare_mod  # noqa: E402
from eval import gate as gate_mod  # noqa: E402
from eval import noise as noise_mod  # noqa: E402
from eval import calibration as calib_mod  # noqa: E402
from eval.runner import DEFAULT_DATASETS, run  # noqa: E402
from portfolio_chatbot.config import list_experiments, load_config  # noqa: E402


def _compare(args) -> int:
    cfg = load_config()
    reports = cfg.resolve(cfg.paths.eval_reports_dir)
    try:
        base = compare_mod.load_report(compare_mod.find_report(reports, args.compare[0], args.dataset))
        cand = compare_mod.load_report(compare_mod.find_report(reports, args.compare[1], args.dataset))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(compare_mod.render(compare_mod.compare(base, cand)))
    return 0


def _calibrate(args) -> int:
    from eval.metrics.judge import build_judge

    try:
        cfg = load_config(args.experiment, overrides=args.overrides)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    judge = build_judge(cfg)
    print(f"calibrating judge: {judge.model}\n")
    progress = None if args.quiet else (lambda line: print(f"  . {line}", flush=True))

    result = calib_mod.calibrate(cfg, judge, progress=progress)
    print()
    print(result.render())
    path = calib_mod.save(result)
    print(f"\nsaved -> {cfg.display_path(path)}")
    return 0 if result.verdict != "unmeasured" else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the chatbot pipeline.")
    p.add_argument("--experiment", "-e", default=None,
                   help=f"Available: {', '.join(list_experiments()) or 'none'}")
    p.add_argument("--dataset", "-d", default="golden_qa")
    p.add_argument("--all", action="store_true", help="Run every dataset.")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--limit", type=int, default=None, help="First N cases only.")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"),
                   help="Compare two previously written reports instead of running.")
    p.add_argument("--repeats", "-r", type=int, default=1, metavar="N",
                   help="Run the dataset N times and report the noise floor "
                        "(sigma and minimum detectable effect per metric). "
                        "Spec default is 5.")
    p.add_argument("--assert", dest="gate", action="store_true",
                   help="Exit non-zero on a blocking gate violation.")
    p.add_argument("--save-baseline", action="store_true",
                   help="Record this run as the regression baseline. With "
                        "--repeats, the measured sigma is stored with it.")
    p.add_argument("--kappa", type=float, default=None,
                   help="Judge agreement. Defaults to the stored calibration "
                        "result; judged metrics cannot block without one.")
    p.add_argument("--judge", action="store_true",
                   help="Score faithfulness and relevancy with the LLM judge. "
                        "Roughly 2 extra calls per turn, on the judge model's "
                        "own quota.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run the judge over eval/datasets/judge_calibration.json "
                        "and report Cohen's kappa. Do this before trusting any "
                        "judged number.")
    args = p.parse_args(argv)

    if args.compare:
        return _compare(args)

    if args.calibrate:
        return _calibrate(args)

    try:
        cfg = load_config(args.experiment, overrides=args.overrides)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    if not cfg.index_path.exists():
        print(f"No index for {cfg.index_fingerprint}. Build it first:", file=sys.stderr)
        name = cfg.run.experiment_name
        print(f"    python scripts/ingest.py"
              + (f" --experiment {name}" if name != "default" else ""), file=sys.stderr)
        return 1

    datasets = DEFAULT_DATASETS if args.all else (args.dataset,)
    progress = None if args.quiet else (lambda line: print(f"  . {line}", flush=True))

    judge = None
    if args.judge or cfg.eval.judge.enabled:
        from eval.metrics.judge import build_judge
        judge = build_judge(cfg)
        print(f"judge: {judge.model}")

    # Kappa comes from the stored calibration unless overridden. Without one,
    # judged metrics demote to warnings - an unvalidated instrument does not get
    # to block a release.
    kappa = args.kappa if args.kappa is not None else calib_mod.current_kappa()
    if judge is not None and kappa is None:
        print("  WARNING: judge is uncalibrated. Judged metrics will report but "
              "cannot block.\n           Run: python scripts/run_eval.py --calibrate")

    failed = 0
    for name in datasets:
        print(f"\n=== {name} ===")
        floor = None
        try:
            if args.repeats > 1:
                # Each repeat gets its own thread namespace, so run 2 never sees
                # run 1's history - the runs must be independent samples.
                floor = noise_mod.measure(
                    lambda i: run(cfg, name, progress=None, limit=args.limit, judge=judge),
                    repeats=args.repeats, dataset=name, progress=progress,
                )
                report = run(cfg, name, progress=None, limit=args.limit, judge=judge)
                print()
                print(floor.render())
            else:
                report = run(cfg, name, progress=progress, limit=args.limit, judge=judge)
        except Exception as exc:
            print(f"Failed: {exc}", file=sys.stderr)
            failed += 1
            continue

        print()
        print(report.render())

        if args.save_baseline:
            path = gate_mod.save_baseline(
                report, sigmas=floor.sigmas if floor else None,
                n_repeats=args.repeats,
            )
            print(f"\nbaseline saved -> {cfg.display_path(path)}")
            if not floor:
                print("  NOTE: no sigma measured. Regression checks will fall back to "
                      "min_absolute.\n  Re-run with --repeats 5 for a real noise floor.")

        # Invariants run on every evaluation, with or without --assert: a leaked
        # URL is not something you should have to opt in to hearing about.
        baseline = gate_mod.load_baseline(name) if args.gate else None
        try:
            result = gate_mod.check(report, baseline, kappa=kappa)
        except FileNotFoundError as exc:
            print(f"\nGate skipped: {exc}", file=sys.stderr)
            continue

        if args.gate or not result.clean:
            print()
            print(result.render())
        if args.gate and not result.passed:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
