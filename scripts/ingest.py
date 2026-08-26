"""
Build the vector index for a config.

    python scripts/ingest.py                                  # default.json
    python scripts/ingest.py --experiment exp002_chunk_512    # an experiment
    python scripts/ingest.py --set ingestion.split.max_chunk_chars=800
    python scripts/ingest.py --dry-run                        # chunk only, no embedding
    python scripts/ingest.py --force                          # rebuild an existing index

The index is keyed by the config's `index_fingerprint`, so re-running for a
config that has already been built is a no-op unless --force is given.
--dry-run skips loading the embedding model entirely, which makes it a fast
way to see how a chunking change lands before paying for embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_chatbot.config import list_experiments, load_config  # noqa: E402
from portfolio_chatbot.ingestion.build_index import build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the FAISS index for a config.")
    p.add_argument("--experiment", "-e", default=None,
                   help=f"Experiment name or path. Available: {', '.join(list_experiments()) or 'none'}")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="Dotted config override, repeatable (e.g. ingestion.split.max_chunk_chars=800)")
    p.add_argument("--force", "-f", action="store_true", help="Rebuild even if the index exists.")
    p.add_argument("--dry-run", "-n", action="store_true",
                   help="Parse and chunk only. No embedding model, no index written.")
    args = p.parse_args(argv)

    try:
        cfg = load_config(args.experiment, overrides=args.overrides)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    try:
        report = build(cfg, force=args.force, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1

    print(report.render())
    if report.skipped and not args.dry_run:
        print("\nUse --force to rebuild.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
