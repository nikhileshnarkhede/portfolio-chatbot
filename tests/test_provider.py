"""
test_provider.py - API key resolution.

Four sources, and the order between them is the whole contract. These exist
because the failure mode is silent: a key file that is never read looks exactly
like no key at all, and the page says "not configured" while sitting on the
key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_chatbot.llm import provider

#: Captured at import, BEFORE the autouse fixture below redirects the constant
#: at every test. Without this the "is it the conventional path" test would
#: assert against the fixture's tmp_path and pass no matter what the real value
#: became - a test that guards a constant has to read it before it is patched.
REAL_SECRET_PATH = provider.DOCKER_SECRET_PATH


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch, tmp_path):
    """Neither the env nor a real /run/secrets may leak into these tests."""
    monkeypatch.delenv(provider.API_KEY_ENV, raising=False)
    monkeypatch.delenv(provider.API_KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(provider, "DOCKER_SECRET_PATH", tmp_path / "absent")


def test_env_var_is_used_when_set(monkeypatch):
    monkeypatch.setenv(provider.API_KEY_ENV, "gsk_from_env")
    assert provider.resolve_api_key() == "gsk_from_env"


def test_key_file_is_read_when_the_env_var_is_absent(monkeypatch, tmp_path):
    keyfile = tmp_path / "groq"
    keyfile.write_text("gsk_from_file", encoding="utf-8")
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))
    assert provider.resolve_api_key() == "gsk_from_file"


def test_the_mounted_secret_path_is_the_last_resort(monkeypatch, tmp_path):
    secret = tmp_path / "groq_api_key"
    secret.write_text("gsk_from_secret", encoding="utf-8")
    monkeypatch.setattr(provider, "DOCKER_SECRET_PATH", secret)
    assert provider.resolve_api_key() == "gsk_from_secret"


def test_env_var_wins_over_both_file_sources(monkeypatch, tmp_path):
    keyfile = tmp_path / "groq"
    keyfile.write_text("gsk_from_file", encoding="utf-8")
    secret = tmp_path / "groq_api_key"
    secret.write_text("gsk_from_secret", encoding="utf-8")

    monkeypatch.setenv(provider.API_KEY_ENV, "gsk_from_env")
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))
    monkeypatch.setattr(provider, "DOCKER_SECRET_PATH", secret)

    assert provider.resolve_api_key() == "gsk_from_env"


def test_key_file_wins_over_the_mounted_secret(monkeypatch, tmp_path):
    keyfile = tmp_path / "groq"
    keyfile.write_text("gsk_from_file", encoding="utf-8")
    secret = tmp_path / "groq_api_key"
    secret.write_text("gsk_from_secret", encoding="utf-8")

    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))
    monkeypatch.setattr(provider, "DOCKER_SECRET_PATH", secret)

    assert provider.resolve_api_key() == "gsk_from_file"


def test_trailing_newline_is_stripped(monkeypatch, tmp_path):
    """`echo gsk_... > key.txt` writes a newline. Groq 401s on key-plus-newline.

    That 401 reads as "your key is invalid", which sends you to the Groq
    console to reissue a key that was fine - so the strip is worth a test.
    """
    keyfile = tmp_path / "groq"
    keyfile.write_text("gsk_real_key\n", encoding="utf-8")
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))
    assert provider.resolve_api_key() == "gsk_real_key"


def test_whitespace_only_env_var_counts_as_absent(monkeypatch, tmp_path):
    """An empty `GROQ_API_KEY=` must not shadow a key file.

    A launcher that always exports the variable sends an unset key through as
    an empty string rather than as no variable at all. Treating that as "set"
    would make the file sources unreachable under the exact setup they exist
    for.
    """
    secret = tmp_path / "groq_api_key"
    secret.write_text("gsk_from_secret", encoding="utf-8")
    monkeypatch.setattr(provider, "DOCKER_SECRET_PATH", secret)
    monkeypatch.setenv(provider.API_KEY_ENV, "   ")

    assert provider.resolve_api_key() == "gsk_from_secret"


def test_a_missing_key_file_falls_through_rather_than_raising(monkeypatch, tmp_path):
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(tmp_path / "nope"))
    assert provider.resolve_api_key() == ""


def test_a_directory_as_the_key_file_falls_through(monkeypatch, tmp_path):
    """Pointing GROQ_API_KEY_FILE at a folder rather than a file.

    Reading a directory raises IsADirectoryError, which is an OSError - so it
    has to fall through to the notice, not escape from inside a node.
    """
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(tmp_path))
    assert provider.resolve_api_key() == ""


def test_no_source_returns_empty_string_and_never_raises():
    assert provider.resolve_api_key() == ""


def test_api_key_raises_and_names_where_to_put_the_key():
    """The error has to name the routes, not just say "not set".

    It is the message a CLI user hits, and every source it omits is one they
    have to go read the code to discover.
    """
    with pytest.raises(provider.MissingAPIKey) as excinfo:
        provider.api_key()

    message = str(excinfo.value)
    assert provider.API_KEY_ENV in message
    assert provider.API_KEY_FILE_ENV in message
    assert "secrets.toml" in message
    assert ".env" in message
    assert "secrets/groq_api_key.txt.example" in message


def test_api_key_returns_what_resolve_found(monkeypatch, tmp_path):
    secret = tmp_path / "groq_api_key"
    secret.write_text("gsk_from_secret\n", encoding="utf-8")
    monkeypatch.setattr(provider, "DOCKER_SECRET_PATH", secret)
    assert provider.api_key() == "gsk_from_secret"


def test_the_mounted_secret_path_is_the_conventional_one():
    """/run/secrets/<name> is where a platform injects a secret file.

    Pinned because the whole value of this path is that it needs no
    configuration - a platform mounts there without being told, so changing it
    silently would break the one source that works with no setup at all.
    """
    assert REAL_SECRET_PATH == Path("/run/secrets/groq_api_key")


# ==========================================================================
# Key-file parsing.
#
# The person filling this file in is often not the person who wrote the code:
# they get secrets/groq_api_key.txt.example, paste their own key, and hand back
# a path. Every shape below is one somebody actually writes, and every one of
# them fails as a Groq 401 reading "your key is invalid" - the single most
# misleading thing the system could say, because it sends them to reissue a key
# that was fine.
# ==========================================================================

@pytest.mark.parametrize("contents, expected", [
    ("gsk_abc123", "gsk_abc123"),
    ("gsk_abc123\n", "gsk_abc123"),
    ("gsk_abc123\r\n", "gsk_abc123"),                      # Windows line endings
    ("  gsk_abc123  ", "gsk_abc123"),
    ("GROQ_API_KEY=gsk_abc123", "gsk_abc123"),
    ("GROQ_API_KEY = gsk_abc123", "gsk_abc123"),
    ('GROQ_API_KEY="gsk_abc123"', "gsk_abc123"),           # nothing strips quotes
    ("GROQ_API_KEY='gsk_abc123'", "gsk_abc123"),
    ("export GROQ_API_KEY=gsk_abc123", "gsk_abc123"),
    ("groq_api_key=gsk_abc123", "gsk_abc123"),             # lowercase
    ("# a comment\n\ngsk_abc123\n", "gsk_abc123"),
    ("﻿gsk_abc123", "gsk_abc123"),                    # Notepad UTF-8 BOM
    ("﻿GROQ_API_KEY=gsk_abc123", "gsk_abc123"),
])
def test_key_file_shapes_all_parse(contents, expected):
    assert provider.parse_key_file(contents) == expected


@pytest.mark.parametrize("contents", [
    "",
    "   \n\n  ",
    "# only comments\n# and more comments\n",
    "GROQ_API_KEY=",
    'GROQ_API_KEY=""',
])
def test_empty_shapes_resolve_to_nothing(contents):
    assert provider.parse_key_file(contents) == ""


def test_the_groq_assignment_wins_over_position():
    """A file holding several secrets must not return whichever is on top."""
    contents = "LANGSMITH_API_KEY=ls_zzz\nOPENAI_API_KEY=sk_yyy\nGROQ_API_KEY=gsk_abc\n"
    assert provider.parse_key_file(contents) == "gsk_abc"


def test_another_secrets_line_is_never_returned_as_the_groq_key():
    """No Groq assignment means no key - not "the first line I found".

    The bare-key fallback would otherwise hand back the literal text
    `LANGSMITH_API_KEY=ls_zzz`, and the 401 that follows would be blamed on
    Groq rather than on the file holding the wrong secret.
    """
    assert provider.parse_key_file("LANGSMITH_API_KEY=ls_zzz\n") == ""


def test_an_empty_assignment_does_not_shadow_a_bare_key_below_it():
    assert provider.parse_key_file("GROQ_API_KEY=\n\ngsk_abc123\n") == "gsk_abc123"


@pytest.mark.parametrize("placeholder", [
    "gsk_replace_this_with_your_own_key",
    "gsk_your_key_here",
    "gsk_...",
    'GROQ_API_KEY="gsk_replace_this_with_your_own_key"',
])
def test_an_unfilled_template_counts_as_no_key(placeholder):
    """An unedited template must say "not configured", not 401.

    A 401 reads as "the key you were given is invalid", which is the one
    conclusion that is false - and it sends the other person back to the Groq
    console instead of to the file they forgot to edit.
    """
    assert provider.parse_key_file(placeholder) == ""


def test_the_shipped_template_parses_to_nothing():
    """The example file must not be mistaken for a usable key.

    Reads the real file rather than a copy of its text: an edit that replaced
    the placeholder with something key-shaped would otherwise ship a template
    that silently resolves.
    """
    template = (
        Path(__file__).resolve().parents[1]
        / "secrets" / "groq_api_key.txt.example"
    )
    assert template.exists(), "secrets/groq_api_key.txt.example is missing"
    assert provider.parse_key_file(template.read_text(encoding="utf-8-sig")) == ""


def test_a_filled_in_copy_of_the_template_resolves(monkeypatch, tmp_path):
    """End to end: the template, edited the way the instructions say to."""
    template = (
        Path(__file__).resolve().parents[1]
        / "secrets" / "groq_api_key.txt.example"
    )
    filled = template.read_text(encoding="utf-8-sig").replace(
        "gsk_replace_this_with_your_own_key", "gsk_a_real_looking_key")

    keyfile = tmp_path / "groq_api_key.txt"
    keyfile.write_text(filled, encoding="utf-8")
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))

    assert provider.resolve_api_key() == "gsk_a_real_looking_key"


def test_a_notepad_bom_file_still_resolves(monkeypatch, tmp_path):
    """Windows Notepad's "UTF-8" writes a BOM. `.strip()` leaves it in place.

    A BOM-prefixed key reaches Groq as a different string and comes back 401,
    from a file that looks perfect in an editor.
    """
    keyfile = tmp_path / "groq_api_key.txt"
    keyfile.write_text("GROQ_API_KEY=gsk_abc123\n", encoding="utf-8-sig")
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))
    assert provider.resolve_api_key() == "gsk_abc123"


@pytest.mark.parametrize("raw, expected", [
    ('"gsk_abc123"', "gsk_abc123"),
    ("'gsk_abc123'", "gsk_abc123"),
    ("  gsk_abc123  ", "gsk_abc123"),
    ("gsk_abc123", "gsk_abc123"),
])
def test_quotes_around_the_env_var_are_stripped(monkeypatch, raw, expected):
    """`--env-file` does not strip quotes, and neither did we.

    `GROQ_API_KEY="gsk_..."` is a completely reasonable thing to write in a
    .env file. Docker passes the quotes through as part of the value, Groq
    returns 401, and the app reports a generic failure - so the one shape of
    this mistake that the key-file parser always handled was still broken on
    the route people actually use with a container.
    """
    monkeypatch.setenv(provider.API_KEY_ENV, raw)
    assert provider.resolve_api_key() == expected


def test_a_placeholder_env_var_falls_through_to_a_real_key_file(monkeypatch, tmp_path):
    """`GROQ_API_KEY=gsk_your_key_here` copied from .env.example is not a key."""
    keyfile = tmp_path / "groq"
    keyfile.write_text("gsk_the_real_one", encoding="utf-8")
    monkeypatch.setenv(provider.API_KEY_ENV, "gsk_your_key_here")
    monkeypatch.setenv(provider.API_KEY_FILE_ENV, str(keyfile))
    assert provider.resolve_api_key() == "gsk_the_real_one"
