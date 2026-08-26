"""
Tests for the Streamlit app, via Streamlit's own AppTest harness.

These run the real script in-process, so they catch the class of failure that
never shows up in a unit test: an exception during page render, which Streamlit
displays as a stack trace where the app should be.

One such bug was found this way. `st.secrets.get()` RAISES
`StreamlitSecretNotFoundError` when no `secrets.toml` exists rather than
returning a default, so the app died on its first line for exactly the person
who most needed to see the "add your API key" message.
`test_app_renders_without_a_secrets_file` is the guard.

These are slower than the rest of the suite (each one boots a Streamlit script
run). Deselect them with `-m "not ui"` while iterating on the pipeline.
"""

from __future__ import annotations

import os
import pytest

pytest.importorskip("streamlit")

from langchain_core.messages import AIMessage  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from portfolio_chatbot.config import PROJECT_ROOT  # noqa: E402

pytestmark = pytest.mark.ui

APP = str(PROJECT_ROOT / "ui" / "app.py")
ALLOWED = "https://github.com/nikhileshnarkhede"
FORGED = "https://evil.example.com"


class FakeLLM:
    def invoke(self, _p):
        return AIMessage(content="earlier turns")

    def stream(self, _p):
        for word in ["I built ", "projects. ", f"See [GitHub]({ALLOWED}) ", f"and [x]({FORGED})."]:
            yield AIMessage(content=word)


@pytest.fixture
def app(cfg, fake_index, monkeypatch):
    """The real app, with a fake index and a fake model."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    for module in ("generate", "summarize"):
        monkeypatch.setattr(f"portfolio_chatbot.nodes.{module}.build_llm",
                            lambda c, model=None: FakeLLM())
    from portfolio_chatbot import graph
    graph.clear_cache()
    return AppTest.from_file(APP, default_timeout=120)


@pytest.fixture
def no_secrets(monkeypatch):
    """Silence BOTH key sources: the environment and secrets.toml."""
    import streamlit.runtime.secrets as st_secrets

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(st_secrets.Secrets, "get",
                        lambda self, key, default=None: None, raising=False)


def test_app_renders_when_secrets_lookup_raises(app, monkeypatch):
    """REGRESSION: st.secrets.get() RAISES when no secrets.toml exists.

    Simulated rather than relying on the file being absent, so the test means
    the same thing on a machine that has one.
    """
    import streamlit.runtime.secrets as st_secrets

    def boom(self, *a, **k):
        raise st_secrets.StreamlitSecretNotFoundError("No secrets found")

    monkeypatch.setattr(st_secrets.Secrets, "get", boom, raising=False)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]


def test_page_shows_title_and_starter_buttons(app, cfg):
    app.run()
    assert app.title[0].value == cfg.ui.heading
    assert [b.label for b in app.button] == [a.label for a in cfg.ui.quick_actions]


def test_a_question_produces_a_sanitized_answer(app):
    app.run()
    app.chat_input[0].set_value("Tell me about your projects").run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert len(app.chat_message) == 2

    reply = " ".join(str(m.value) for m in app.chat_message[-1].markdown)
    assert FORGED not in reply, "the guard must run before the final paint"
    assert ALLOWED in reply


def test_starter_buttons_disappear_once_a_question_is_asked(app):
    app.run()
    app.chat_input[0].set_value("Tell me about your projects").run()
    assert [b.label for b in app.button] == []


def test_missing_api_key_shows_a_message_not_a_crash(cfg, fake_index, no_secrets):
    """Both key sources must be silenced, not just the environment.

    An earlier version only deleted GROQ_API_KEY from the environment. That
    passed on a machine with no secrets.toml and failed on every machine that
    had one, because the app reads the key from secrets and puts it BACK into
    the environment. A test whose result depends on whether the developer has
    configured their own credentials is not testing the app.
    """
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert any("API key" in str(e.value) for e in at.error)


def test_app_runs_under_streamlits_own_sys_path(tmp_path):
    """REGRESSION: `streamlit run ui/app.py` puts ui/ on sys.path, NOT the root.

    This shipped broken once with a green suite, because pytest.ini had `.` on
    the path and the app never had to bootstrap its own. Run in a subprocess,
    from a different working directory, with only the script's own directory
    visible - which is what Streamlit actually does.
    """
    import subprocess
    import sys as _sys

    script = f"""
import sys, os
sys.path = [p for p in sys.path if p not in ('', os.getcwd())]
sys.path.insert(0, r{str(PROJECT_ROOT / 'ui')!r})
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(r{APP!r}, default_timeout=90)
at.run()
print("EXCEPTIONS", len(at.exception))
for e in at.exception:
    print("DETAIL", str(e.value)[:200])
"""
    result = subprocess.run(
        [_sys.executable, "-c", script], cwd=str(tmp_path),
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": "", "GROQ_API_KEY": "test-key"},
    )
    assert "EXCEPTIONS 0" in result.stdout, result.stdout + result.stderr
