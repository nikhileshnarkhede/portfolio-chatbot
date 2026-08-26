"""
Tests for the URL allowlist.

The guard is the last line of defence against a fabricated link reaching a
recruiter, and it is the one component whose failure would be both silent and
damaging - a plausible-looking but wrong GitHub URL in an answer is worse than
no URL at all.

It is also the source of the `attempted_forged_url` metric, which is only
measurable because the guard reports what it removed rather than stripping
silently the way the original app.py did.
"""

from __future__ import annotations


from portfolio_chatbot.config import load_config
from portfolio_chatbot.nodes.sanitize import sanitize
from portfolio_chatbot.state import new_turn
from portfolio_chatbot.tools.link_guard import sanitize as strip_links

ALLOWED = "https://github.com/nikhileshnarkhede"
FORGED = "https://github.com/nikhileshnarkhede-fake"


def state_for(question: str, **extra):
    s = new_turn(question, experiment_name="t", run_fingerprint="r", index_fingerprint="i")
    s.update(extra)
    return s


# ---------------------------------------------------------------- link guard

def test_allowed_markdown_link_survives(cfg):
    text = f"See [my GitHub]({ALLOWED})."
    clean, audit = strip_links(text, cfg)
    assert ALLOWED in clean
    assert audit["stripped"] == []


def test_forged_link_is_stripped_but_label_survives(cfg):
    clean, audit = strip_links(f"See [my GitHub]({FORGED}).", cfg)
    assert FORGED not in clean
    assert "my GitHub" in clean
    assert audit["stripped"] == [FORGED]


def test_bare_disallowed_url_is_removed(cfg):
    clean, audit = strip_links(f"Visit {FORGED} now", cfg)
    assert FORGED not in clean
    assert audit["stripped_count"] == 1


def test_bare_allowed_url_survives(cfg):
    clean, audit = strip_links(f"Visit {ALLOWED} now", cfg)
    assert ALLOWED in clean
    assert audit["stripped"] == []


def test_trailing_slash_is_ignored_when_configured(cfg):
    clean, audit = strip_links(f"[x]({ALLOWED}/)", cfg)
    assert audit["stripped"] == []


def test_repeated_bad_url_is_reported_once_but_counted_twice(cfg):
    _, audit = strip_links(f"[a]({FORGED}) and [b]({FORGED})", cfg)
    assert audit["stripped"] == [FORGED]
    assert audit["stripped_count"] == 2


def test_guard_can_be_disabled(cfg):
    off = load_config(overrides=["safety.url_allowlist.enabled=false"])
    clean, audit = strip_links(f"[x]({FORGED})", off)
    assert FORGED in clean
    assert audit["stripped_count"] == 0


def test_sanitize_node_keeps_draft_and_writes_answer(runnable_config):
    out = sanitize(state_for("q", draft_answer=f"See [x]({FORGED}) and [y]({ALLOWED})"),
                   runnable_config)
    assert FORGED not in out["answer"]
    assert ALLOWED in out["answer"]
    assert out["link_audit"]["stripped_count"] == 1

# ---------------------------------------------------------------- edge cases

def test_multiple_forged_links_all_stripped(cfg):
    text = f"[a]({FORGED}) then [b](https://evil.example.com) then [c]({ALLOWED})"
    clean, audit = strip_links(text, cfg)
    assert audit["stripped_count"] == 2
    assert ALLOWED in clean


def test_answer_without_links_is_untouched(cfg):
    text = "I built eleven projects across ML, NLP and computer vision."
    clean, audit = strip_links(text, cfg)
    assert clean == text
    assert audit["stripped"] == []


def test_stripping_does_not_leave_double_spaces(cfg):
    clean, _ = strip_links(f"Visit {FORGED} for details", cfg)
    assert "  " not in clean


def test_every_allowlisted_url_survives(cfg):
    """The allowlist must not drift from what the resume actually contains."""
    for url in cfg.safety.url_allowlist.urls:
        clean, audit = strip_links(f"see [link]({url})", cfg)
        assert audit["stripped"] == [], f"allowlisted URL was stripped: {url}"


def test_guard_runs_even_on_an_error_reply(cfg, runnable_config):
    """A guard that only runs on the happy path is not a guard."""
    out = sanitize(state_for("q", draft_answer=f"Something failed, see {FORGED}"),
                   runnable_config)
    assert FORGED not in out["answer"]
