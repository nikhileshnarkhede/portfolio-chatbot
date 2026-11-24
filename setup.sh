#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command local setup for macOS / Linux.
#
#   ./setup.sh          (or: bash setup.sh)
#
# What it does, and it is safe to run again at any time:
#   1. creates a Python virtual environment in .venv/
#   2. installs the pinned requirements
#   3. asks for your Groq API key once and saves it in .streamlit/secrets.toml
#      (that file is gitignored, so the key can never be committed)
#   4. starts the chatbot at http://localhost:8501
#
# The first start builds the search index from data/raw/resume.txt, which
# takes about a minute. Later starts are instant until you edit the resume.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

SECRETS=".streamlit/secrets.toml"
RESUME="data/raw/resume.txt"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[!] %s\033[0m\n' "$*"; }

# ---- 1. Python ------------------------------------------------------------
# runtime.txt pins 3.12 and requirements.txt was resolved against it.
PY=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python was not found. Install Python 3.12 from https://www.python.org/downloads/ and run this again."
    exit 1
fi
VER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
if [ "$VER" != "3.12" ]; then
    warn "Found Python $VER. This project is tested on Python 3.12; if installing fails, install 3.12 and re-run."
fi

if [ ! -d .venv ]; then
    say "Creating virtual environment (.venv) with Python $VER"
    "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

# ---- 2. Requirements --------------------------------------------------------
say "Installing requirements (first time takes a few minutes)"
"$VPY" -m pip install --upgrade pip --quiet
"$VPY" -m pip install -r requirements.txt --quiet

# ---- 3. Resume ------------------------------------------------------------
if [ ! -f "$RESUME" ]; then
    warn "No $RESUME found. Copying the template; edit it with your own details."
    cp data/raw/resume.example.txt "$RESUME"
elif command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1 \
     && git diff --quiet HEAD -- "$RESUME" 2>/dev/null; then
    warn "$RESUME is still the original author's resume."
    warn "Replace it with yours (start from data/raw/resume.example.txt), then re-run."
fi

# ---- 4. API key -------------------------------------------------------------
if [ ! -f "$SECRETS" ]; then
    say "Groq API key"
    echo "Create a free key at https://console.groq.com/keys (it starts with gsk_)."
    KEY=""
    while [ -z "$KEY" ]; do
        read -r -s -p "Paste your key and press Enter (input is hidden): " KEY
        echo
    done
    mkdir -p .streamlit
    umask 077
    {
        echo "# Created by setup.sh. Gitignored - never commit this file."
        echo "GROQ_API_KEY = \"$KEY\""
        echo "EXPERIMENT = \"my_profile\""
    } > "$SECRETS"
    echo "Saved to $SECRETS"
else
    echo "Using existing $SECRETS"
fi

# ---- 5. Run -----------------------------------------------------------------
say "Starting the chatbot at http://localhost:8501  (Ctrl+C to stop)"
exec "$VPY" -m streamlit run ui/app.py
