"""
app.py - the Streamlit entry point.

    streamlit run ui/app.py

Deliberately thin. It loads config, wires secrets into the environment, draws
the chrome, and hands each question to the compiled graph. There is no
retrieval logic, no prompt text, and no model call in this file or anywhere
under `ui/` - that separation is what lets an eval run exercise the identical
pipeline with no display attached.

The original app.py was 1129 lines with all six concerns interleaved. This is
the same product with the pipeline lifted out.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

# `streamlit run ui/app.py` puts the SCRIPT'S directory (ui/) on sys.path - not
# the project root, and not the working directory. So both the project root
# (for `ui.components`) and src/ (for `portfolio_chatbot`) have to be added
# here, before any first-party import.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from portfolio_chatbot.config import load_config  # noqa: E402
from portfolio_chatbot.llm.provider import (  # noqa: E402
    API_KEY_ENV,
    API_KEY_FILE_ENV,
    resolve_api_key,
)
from ui.components import chat, debug_panel, header, sidebar_controls  # noqa: E402


def _secret(name: str, default: str | None = None) -> str | None:
    """Read one Streamlit secret, tolerating the absence of secrets.toml.

    `st.secrets` RAISES `StreamlitSecretNotFoundError` when no secrets file
    exists - it does not return empty. Without this guard the app dies on the
    first line for anyone who has not created the file yet, which is exactly
    the person who most needs to see the "add your API key" message.
    """
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    # Environment fallback so `python scripts/serve.py --experiment X` works
    # without writing the choice into secrets.toml.
    return value if value is not None else os.environ.get(name, default)


def _load_secrets() -> bool:
    """Move Streamlit secrets into the environment.

    `llm/provider.py` reads `GROQ_API_KEY` from the environment so the CLI
    entrypoints and the app share one code path, and so the key never lands in
    the resolved config that gets written into `runs/`.

    secrets.toml wins when present, then `resolve_api_key()` covers the rest -
    the env var, `GROQ_API_KEY_FILE`, and a mounted secret file. Without that
    second call the page would report "no key configured" while sitting on a
    perfectly good key file.
    """
    key = _secret(API_KEY_ENV, "") or resolve_api_key()
    if key:
        os.environ[API_KEY_ENV] = str(key).strip()
    return bool(os.environ.get(API_KEY_ENV))


def _missing_key_notice() -> None:
    """What somebody sees when no key reached the app.

    Written for a stranger with their own Groq key, because that is who reads
    it: the three steps, in order, with nothing to work out. The routes that
    keep the key out of a shell history are in the expander below, not in the
    way of somebody who just wants the app running.
    """
    st.error("No Groq API key configured.")
    st.markdown(
        "This chatbot needs a **Groq API key** — it's free, and it stays yours. "
        "The key is read at startup and is never committed or logged."
    )

    st.markdown("**1.** Get a key at [console.groq.com/keys]"
                "(https://console.groq.com/keys) — it starts with `gsk_`.")
    st.markdown("**2.** Put it in `.streamlit/secrets.toml`:")
    st.code(f'{API_KEY_ENV} = "gsk_your_actual_key"', language="toml")
    st.markdown("**3.** Restart the app:")
    st.code("streamlit run ui/app.py", language="bash")

    with st.expander("Other ways to pass the key"):
        st.markdown("**An environment variable**, which the CLI entrypoints "
                    "(`scripts/ingest.py`, `scripts/run_eval.py`) read too:")
        st.code(
            f"# macOS / Linux\nexport {API_KEY_ENV}=gsk_...\n\n"
            f"# Windows PowerShell\n$env:{API_KEY_ENV} = \"gsk_...\"",
            language="bash",
        )
        st.markdown("**A key file**, so the key is never in your shell history "
                    "or in the environment of every process you launch:")
        st.code(
            f"# macOS / Linux\nexport {API_KEY_FILE_ENV}=/path/to/groq_api_key.txt\n\n"
            f"# Windows PowerShell\n$env:{API_KEY_FILE_ENV} = \"C:\\keys\\groq_api_key.txt\"",
            language="bash",
        )
        st.markdown(
            "Copy `secrets/groq_api_key.txt.example` to start from a template. "
            "The file can hold a bare key, `GROQ_API_KEY=...`, or a quoted "
            "value, with comments and blank lines around it — all three parse. "
            "A template left unedited resolves to *no key*, so you get this "
            "page rather than a 401 claiming your key is invalid."
        )
        st.markdown(
            "**On Streamlit Community Cloud**, put the key in the app's own "
            "Secrets settings rather than in the repository."
        )


@st.cache_resource(show_spinner=False)
def _build_index(fingerprint: str, _cfg) -> bool:
    """Build the index if this deployment has not got one. True if usable.

    `data/index/` is gitignored - it is derived data, and one directory per
    experiment fingerprint adds up fast - so a fresh Streamlit Cloud deploy
    arrives with the resume but no vectors, and there is no build hook to run
    `scripts/ingest.py` in. Building on first use is what closes that gap.

    Committing the index instead would look like the cheaper fix and mostly is
    not: the app has to embed the QUESTION on every turn, so MiniLM is
    downloaded either way. Shipping vectors would save embedding 61 chunks -
    seconds - while making derived data a thing that can drift from the resume
    it was built from.

    Cached on the FINGERPRINT, not on the config object: it is the identity of
    the index, so a config change rebuilds and a rerun does not. `_cfg` is
    underscore-prefixed so Streamlit does not try to hash a pydantic model.
    """
    from portfolio_chatbot.ingestion.build_index import build

    with st.spinner("Building the search index — first run only, about a minute…"):
        try:
            report = build(_cfg)
        except Exception as exc:
            st.error(f"Could not build the index for `{fingerprint}`.")
            st.code(str(exc), language="text")
            st.caption("Locally, `python scripts/ingest.py` does the same thing "
                       "with the full error.")
            return False

    if not report.skipped:
        st.success(f"Indexed {report.n_chunks} sections from the resume.")
    return True


def main() -> None:
    base = load_config(_secret("EXPERIMENT"))

    st.set_page_config(page_title=base.ui.page_title, page_icon=base.ui.page_icon)
    header.inject_styles()

    chat.init_state()
    st.session_state.setdefault("thread_id", uuid.uuid4().hex[:12])

    cfg = base
    if base.ui.show_experiment_switcher:
        cfg, changed = sidebar_controls.render(base)
        if changed:
            chat.reset(uuid.uuid4().hex[:12])
            st.rerun()

    header.nav(cfg)
    header.hero(cfg)
    header.about(cfg)

    has_key = _load_secrets()

    if not cfg.index_path.exists() and not _build_index(cfg.index_fingerprint, cfg):
        header.footer(cfg)
        return

    # Read the question BEFORE drawing the starter buttons, so they disappear
    # on the same rerun that answers the first question rather than lingering
    # above the reply until the next interaction. `st.chat_input` is pinned to
    # the bottom of the page by Streamlit regardless of call order.
    question = chat.take_question(cfg)
    if not question:
        chat.quick_actions(cfg)

    chat.render_history(cfg)

    if question:
        if not has_key:
            _missing_key_notice()
        else:
            chat.answer(cfg, question, st.session_state.thread_id)

    if st.session_state.get("show_debug", cfg.ui.show_debug):
        st.divider()
        st.subheader("Last turn")
        debug_panel.render(cfg, st.session_state.last_state)

    header.footer(cfg)


if __name__ == "__main__":
    main()
