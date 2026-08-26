"""
Launch the Streamlit app, after checking it can actually run.

    python scripts/serve.py
    python scripts/serve.py --experiment exp003_prompt_v2
    python scripts/serve.py --port 8502

`streamlit run ui/app.py` works fine on its own. This wrapper exists to fail
fast with a readable message instead of a rendered stack trace: it verifies the
config parses, the prompts load, the index exists and an API key is present -
all before Streamlit boots.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from portfolio_chatbot.config import list_experiments, load_config  # noqa: E402
from portfolio_chatbot.llm.provider import API_KEY_ENV  # noqa: E402
from portfolio_chatbot.prompts.loader import active_prompts  # noqa: E402


def preflight(experiment: str | None) -> tuple[bool, list[str]]:
    problems: list[str] = []

    try:
        cfg = load_config(experiment)
    except Exception as exc:
        return False, [f"config: {exc}"]

    try:
        prompts = active_prompts(cfg)
        print(f"  prompts   {', '.join(f'{k}={v.prompt_id}' for k, v in prompts.items())}")
    except Exception as exc:
        problems.append(f"prompts: {exc}")

    if cfg.index_path.exists():
        print(f"  index     {cfg.index_fingerprint} ready")
    else:
        name = cfg.run.experiment_name
        problems.append(
            f"no index for {cfg.index_fingerprint}. Run: python scripts/ingest.py"
            + (f" --experiment {name}" if name != "default" else "")
        )

    secrets = ROOT / ".streamlit" / "secrets.toml"
    if os.environ.get(API_KEY_ENV) or secrets.exists():
        print("  api key   present")
    else:
        problems.append(
            f"no {API_KEY_ENV}. Copy .streamlit/secrets.toml.example to "
            f".streamlit/secrets.toml, or export it."
        )

    print(f"  model     {cfg.llm.primary_model} (temp {cfg.llm.temperature})")
    return not problems, problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Preflight and launch the Streamlit app.")
    p.add_argument("--experiment", "-e", default=None,
                   help=f"Available: {', '.join(list_experiments()) or 'none'}")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--check", action="store_true", help="Preflight only; do not launch.")
    args = p.parse_args(argv)

    print("preflight:")
    ok, problems = preflight(args.experiment)
    if not ok:
        print("\ncannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if args.check:
        print("\nall checks passed.")
        return 0

    env = dict(os.environ)
    if args.experiment:
        # ui/app.py reads EXPERIMENT from Streamlit secrets; the env var is the
        # CLI equivalent and is picked up the same way.
        env["EXPERIMENT"] = args.experiment

    print(f"\nstarting on port {args.port} ...\n")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"),
         "--server.port", str(args.port)],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
