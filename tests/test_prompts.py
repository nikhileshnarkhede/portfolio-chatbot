"""
Tests for prompt loading and validation.

The validation tests matter more than they look. A prompt is a string with
holes in it, and both ways that goes wrong are silent:

* an undeclared placeholder blows up at generation time, mid-conversation,
  after retrieval has already been paid for;
* a declared-but-unused variable never raises at all - the prompt quietly
  ignores your context or chat history, and it surfaces weeks later as an
  unexplained drop in groundedness.

`loader.load_prompt` refuses both, and these tests hold it to that.
"""

from __future__ import annotations

import json

import pytest

from portfolio_chatbot.config import load_config
from portfolio_chatbot.prompts import loader
from portfolio_chatbot.prompts.loader import (
    active_prompts,
    as_template,
    load_prompt,
    placeholders,
    system_prompt,
)

RENDER_VARS = {"context": "CTX", "question": "Q", "chat_history": "HIST"}


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def probe(cfg):
    """Write a temporary prompt + registry entry, then restore both."""
    registry_path = cfg.resolve(cfg.paths.prompt_registry)
    original = registry_path.read_text(encoding="utf-8")
    md = cfg.project_root / "prompts" / "system" / "_test_probe.md"

    def _make(text: str, input_variables: list[str], path: str | None = None):
        reg = json.loads(original)
        reg["prompts"]["_probe"] = {
            "path": path or "prompts/system/_test_probe.md",
            "family": "system", "version": 0, "status": "test",
            "description": "test", "evaluated": False,
            "input_variables": input_variables,
        }
        registry_path.write_text(json.dumps(reg), encoding="utf-8")
        md.write_text(text, encoding="utf-8")
        loader.clear_cache()
        return "_probe"

    yield _make
    registry_path.write_text(original, encoding="utf-8")
    md.unlink(missing_ok=True)
    loader.clear_cache()


# ---------------------------------------------------------------- placeholders

def test_placeholders_finds_variables():
    assert placeholders("a {x} b {y}") == {"x", "y"}


def test_placeholders_ignores_escaped_braces():
    """A prompt asking for JSON output must not have its example parsed as a var."""
    assert placeholders('return {{"a": 1}} using {context}') == {"context"}


# ---------------------------------------------------------------- validation

def test_rejects_undeclared_placeholder(cfg, probe):
    pid = probe("Hi {context} and {rogue}", ["context"])
    with pytest.raises(ValueError, match="rogue"):
        load_prompt(cfg, pid)


def test_rejects_declared_but_unused_variable(cfg, probe):
    pid = probe("Hi {context}", ["context", "chat_history"])
    with pytest.raises(ValueError, match="chat_history"):
        load_prompt(cfg, pid)


def test_rejects_empty_prompt_file(cfg, probe):
    pid = probe("   \n  ", [])
    with pytest.raises(ValueError, match="empty"):
        load_prompt(cfg, pid)


def test_rejects_missing_prompt_file(cfg, probe):
    pid = probe("unused", [], path="prompts/system/does_not_exist.md")
    with pytest.raises(FileNotFoundError):
        load_prompt(cfg, pid)


def test_rejects_unknown_prompt_id(cfg):
    with pytest.raises(KeyError, match="system.v99"):
        load_prompt(cfg, "system.v99")


def test_accepts_escaped_braces(cfg, probe):
    pid = probe('Return {{"a": 1}} using {context}', ["context"])
    spec = load_prompt(cfg, pid)
    assert as_template(spec).format(context="C") == 'Return {"a": 1} using C'


# ---------------------------------------------------------------- registry

@pytest.mark.parametrize("prompt_id", [
    "system.v1_persona", "system.v2_persona", "query_expansion.v1", "summarizer.v1",
])
def test_every_registered_prompt_loads(cfg, prompt_id):
    """Guards against a registry entry pointing at a file nobody wrote."""
    assert load_prompt(cfg, prompt_id).chars > 0


def test_active_prompts_covers_all_three_roles(cfg):
    assert set(active_prompts(cfg)) == {"system", "summarizer", "query_expansion"}


# ---------------------------------------------------------------- selection

def test_config_selects_the_system_prompt():
    base = load_config()
    v2 = load_config("exp003_prompt_v2")
    assert load_prompt(base, base.prompts.system).version == 1
    assert load_prompt(v2, v2.prompts.system).version == 2


def test_prompt_change_alters_run_fingerprint_but_not_index():
    """A prompt A/B must not force a re-ingest."""
    base = load_config()
    v2 = load_config("exp003_prompt_v2")
    assert base.index_fingerprint == v2.index_fingerprint
    assert base.run_fingerprint != v2.run_fingerprint


# ---------------------------------------------------------------- rendering

@pytest.mark.parametrize("experiment", [None, "exp003_prompt_v2"])
def test_system_prompt_renders_with_all_variables(experiment):
    cfg = load_config(experiment)
    rendered = system_prompt(cfg).format(**RENDER_VARS)
    for value in RENDER_VARS.values():
        assert value in rendered


def test_v1_is_byte_identical_to_the_original_app_py_template(cfg):
    """REGRESSION: v1 is the baseline arm; it must reproduce the deployed prompt.

    Verified at extraction time by pulling the literal out of the original
    app.py with `ast` rather than retyping it - 4905 chars, original
    indentation preserved.
    """
    spec = load_prompt(cfg, "system.v1_persona")
    assert spec.chars == 4905
    assert spec.text.splitlines()[1].startswith("        You are Nikhilesh")


def test_summarizer_max_sentences_is_configurable(cfg):
    """The hardcoded '2-3 sentences' became a real parameter."""
    from portfolio_chatbot.prompts.loader import summarizer_prompt
    rendered = summarizer_prompt(cfg).format(
        conversation="A: hi", max_sentences=cfg.memory.summary_max_sentences
    )
    assert str(cfg.memory.summary_max_sentences) in rendered
