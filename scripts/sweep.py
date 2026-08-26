"""
Sweep one parameter across values and report the trend.

    python scripts/sweep.py --set retrieval.filtered.k --values 6 10 14 20
    python scripts/sweep.py --set ingestion.split.max_chunk_chars --values 512 800 1300 --ingest

Each value becomes its own config, its own run, and its own row. A sweep over
an INGESTION parameter changes the index fingerprint, so those need --ingest to
build each index first; retrieval and llm parameters reuse the existing one.

The point of a sweep over a single-arm A/B is that it shows the shape of the
response. One config beating another tells you very little when the difference
is within noise; a monotone trend across four values tells you something real.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval.runner import run  # noqa: E402
from portfolio_chatbot.config import load_config  # noqa: E402
from portfolio_chatbot.ingestion.build_index import build  # noqa: E402

COLUMNS = ["route_correct", "hit_at_k", "recall_at_k", "mrr", "type_precision",
           "fact_coverage", "attempted_forged_url", "latency_mean"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sweep one config parameter.")
    p.add_argument("--set", dest="key", required=True, help="Dotted config path.")
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--experiment", "-e", default=None, help="Base config for the sweep.")
    p.add_argument("--dataset", "-d", default="golden_qa")
    p.add_argument("--ingest", action="store_true",
                   help="Build the index for each value first (needed for ingestion.* keys).")
    args = p.parse_args(argv)

    rows = []
    for value in args.values:
        override = f"{args.key}={value}"
        print(f"\n=== {override} ===", flush=True)
        try:
            cfg = load_config(args.experiment, overrides=[override])
        except Exception as exc:
            print(f"  config error: {exc}", file=sys.stderr)
            continue

        if args.ingest and not cfg.index_path.exists():
            print(f"  building index {cfg.index_fingerprint} ...", flush=True)
            build(cfg)

        if not cfg.index_path.exists():
            print(f"  no index for {cfg.index_fingerprint}; pass --ingest", file=sys.stderr)
            continue

        report = run(cfg, args.dataset, progress=None)
        rows.append((value, report.summary))
        print(f"  done: {report.n_turns} turns, {report.n_errors} errors")

    if not rows:
        print("No successful runs.", file=sys.stderr)
        return 1

    width = max(len(str(v)) for v, _ in rows) + 2
    print(f"\n{args.key}\n")
    print(f"{'value':<{width}}" + "".join(f"{c[:13]:>15}" for c in COLUMNS))
    print("-" * (width + 15 * len(COLUMNS)))
    for value, summary in rows:
        cells = []
        for column in COLUMNS:
            v = summary.get(column)
            try:
                f = float(v)
                cells.append("            n/a" if f != f else f"{f:>15.3f}")
            except (TypeError, ValueError):
                cells.append(f"{'n/a':>15}")
        print(f"{value:<{width}}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
