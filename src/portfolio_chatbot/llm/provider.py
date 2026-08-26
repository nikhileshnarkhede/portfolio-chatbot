"""
provider.py - the chat model factory.

One place builds an LLM, cached per (model, sampling settings) so a Streamlit
rerun or an eval loop does not rebuild a client per turn.

The API key is read from the environment, never from the config. `.env` and
`.streamlit/secrets.toml` both end up as `GROQ_API_KEY`; the UI is responsible
for exporting it before the graph runs. That keeps secrets out of
`resolved_config.json`, which is written into every run directory and would
otherwise carry the key into `runs/` and any report you shared.

**Four sources, in order.** The last two take the key as a FILE rather than a
value:

1. `GROQ_API_KEY` - the environment variable. `.env` covers this for the CLI
   entrypoints; the app moves secrets.toml into it before the graph runs.
2. `GROQ_API_KEY_FILE` - a path to read the key from. A key you export lands in
   your shell history and in the environment of every process that shell
   launches; a key in a file does neither, and it survives closing the terminal.
3. `/run/secrets/groq_api_key` - the conventional mount point for an injected
   secret file. Nothing in this repository mounts anything there today (the
   Docker setup was removed), but it costs one `read_text` on a path that
   normally does not exist, and it is the one location a container platform
   would use without being told.

**The key file is parsed forgivingly, and that is deliberate.** The person
filling it in is often not the person who wrote this code - they get a template,
paste their own key, and hand back a path. Every one of these is accepted:

    gsk_abc123
    GROQ_API_KEY=gsk_abc123
    GROQ_API_KEY = "gsk_abc123"
    export GROQ_API_KEY='gsk_abc123'
    # a comment, then a blank line, then any of the above

because each is what somebody genuinely writes into a file called
`groq_api_key.txt`, and every one of them fails as a 401 that reads "your key is
invalid" rather than as "your file has quotes in it". A parser that only
accepted the bare form would spend other people's afternoons.

Three specifics worth naming, all of them regression-tested:

* **BOM.** Windows Notepad saving as UTF-8 prepends `﻿`, which `.strip()`
  does not remove and Groq rejects. Read as `utf-8-sig`.
* **Quotes.** Nothing else strips them, so `GROQ_API_KEY="gsk_..."` otherwise
  reaches the API with the quote characters attached.
* **Which line.** A file holding several secrets is scanned for a
  `GROQ_API_KEY=` assignment first; only a file with no assignment at all is
  read as a bare key. Taking line 1 blindly hands back whichever secret happens
  to sit at the top.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from ..config import AppConfig

API_KEY_ENV = "GROQ_API_KEY"
API_KEY_FILE_ENV = "GROQ_API_KEY_FILE"
DOCKER_SECRET_PATH = Path("/run/secrets/groq_api_key")


class MissingAPIKey(RuntimeError):
    """Raised when no API key is available for the configured provider."""


#: `GROQ_API_KEY=value`, with optional `export`, spaces and case variations.
_ASSIGNMENT_RE = re.compile(
    rf"^\s*(?:export\s+)?{API_KEY_ENV}\s*=\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)

#: Any `NAME=` line. Used to recognise a line as SOMEBODY's assignment so the
#: bare-key fallback never returns another secret's line as the Groq key.
_ANY_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=")

#: Substrings marking a template that was mounted without being filled in.
#: Sending one of these to Groq returns 401, which reads as "the key you were
#: given is invalid" - the one conclusion that is false. Treated as absent
#: instead, so the app renders "no key configured" and points at the file.
_PLACEHOLDER_MARKERS = ("replace_this", "your_key_here", "your_own_key", "...")


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def _unquote(value: str) -> str:
    """Drop one matching pair of surrounding quotes.

    `--env-file` does not do this, so `GROQ_API_KEY="gsk_..."` otherwise reaches
    Groq with the quote characters still attached.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].strip()
    return value


def parse_key_file(text: str) -> str:
    """The key inside a key file's contents, in whatever shape it was written.

    Returns "" when there is nothing usable, so an empty or comment-only file
    falls through to the next source rather than returning blank as if it were
    an answer.
    """
    # `_read` decodes as utf-8-sig so a Notepad BOM is already gone by the time
    # it gets here - but stripping it again costs nothing and makes this
    # function correct on its own, for any caller that decoded as plain utf-8.
    lines = text.lstrip("﻿").splitlines()

    # An explicit assignment wins wherever it sits, so a file holding several
    # secrets does not return whichever one is on top.
    for line in lines:
        match = _ASSIGNMENT_RE.match(line)
        if match:
            value = _unquote(match.group("value"))
            if value and not _is_placeholder(value):
                return value

    # No usable assignment: treat the file as a bare key and take the first
    # line with content on it - but never a line that is itself an assignment.
    # Without that guard a file holding only `LANGSMITH_API_KEY=ls_...` hands
    # back the whole line as the Groq key, and an empty `GROQ_API_KEY=` above a
    # real bare key hands back the literal text `GROQ_API_KEY=`.
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _ANY_ASSIGNMENT_RE.match(stripped):
            continue
        value = _unquote(stripped)
        if value and not _is_placeholder(value):
            return value

    return ""


def _read(path: Path) -> str:
    """The key in `path`, or "" if it cannot be read.

    Unreadable is treated as absent on purpose: a wrong path, a bad mount or a
    permission problem should fall through to the next source and end at the
    one message that lists every option, not surface as an OSError from inside
    a node.

    `utf-8-sig` rather than `utf-8`: Notepad's "UTF-8" writes a BOM, `.strip()`
    leaves it in place, and Groq 401s on the BOM-prefixed key.
    """
    try:
        return parse_key_file(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError):
        return ""


def resolve_api_key() -> str:
    """The key from the first source that has one, or "" if none do.

    Never raises. `api_key()` is the raising variant; the UI uses this one to
    decide whether to render its own notice.
    """
    # `_unquote`, not just `.strip()`. `docker run --env-file .env` does NOT
    # strip quotes, so a perfectly reasonable `GROQ_API_KEY="gsk_..."` in a
    # .env file arrives here as a value with the quote characters still in it,
    # goes to Groq verbatim, and comes back 401. The key file parser handled
    # this from the start; the environment variable did not, which made the
    # .env route the one shape of this mistake that still broke.
    key = _unquote(os.environ.get(API_KEY_ENV, ""))
    if key and not _is_placeholder(key):
        return key

    path = os.environ.get(API_KEY_FILE_ENV, "").strip()
    if path:
        key = _read(Path(path))
        if key:
            return key

    return _read(DOCKER_SECRET_PATH)


def api_key() -> str:
    key = resolve_api_key()
    if not key:
        raise MissingAPIKey(
            f"{API_KEY_ENV} is not set. Supply it one of four ways: the "
            f"{API_KEY_ENV} environment variable, `.streamlit/secrets.toml` "
            f"for the app, `.env` for the CLI entrypoints, or "
            f"{API_KEY_FILE_ENV} pointing at a file holding the key "
            f"(see secrets/groq_api_key.txt.example)."
        )
    return key


@lru_cache(maxsize=8)
def _build(provider: str, model: str, temperature: float, max_tokens: int | None,
           top_p: float, timeout: int, max_retries: int, key: str) -> BaseChatModel:
    if provider != "groq":  # pragma: no cover - config validation catches this
        raise ValueError(f"Unsupported LLM provider: {provider!r}")

    from langchain_groq import ChatGroq

    kwargs: dict = {
        "model": model,
        "api_key": key,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    # Passing these as None overrides Groq's own defaults on some versions, so
    # only send them when the config actually sets a value.
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None and top_p != 1.0:
        kwargs["top_p"] = top_p

    return ChatGroq(**kwargs)


def build_llm(cfg: AppConfig, model: str | None = None) -> BaseChatModel:
    """The chat model for `model` (default: the config's primary)."""
    llm = cfg.llm
    return _build(
        llm.provider, model or llm.primary_model, llm.temperature,
        llm.max_tokens, llm.top_p, llm.request_timeout, llm.max_retries, api_key(),
    )


def clear_cache() -> None:
    _build.cache_clear()


__all__ = [
    "build_llm", "api_key", "resolve_api_key", "parse_key_file", "clear_cache",
    "MissingAPIKey", "API_KEY_ENV", "API_KEY_FILE_ENV", "DOCKER_SECRET_PATH",
]
