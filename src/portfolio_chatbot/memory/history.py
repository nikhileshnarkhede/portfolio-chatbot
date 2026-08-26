"""
history.py - rendering conversation history as prompt text.

Two different renderings, which is why this is its own module rather than a
helper hidden inside a node:

* `for_prompt` - what `generate` puts in `{chat_history}`: the rolling summary
  of dropped turns, then the recent messages still in the window.
* `for_summary` - what `summarize` condenses: only the turns about to be
  dropped, with no summary header, so the model is not asked to summarize a
  summary.

Both label speakers using `memory.user_label` / `memory.assistant_label`
("Recruiter" / "Nikhilesh"), which the persona prompt depends on - the model is
told it *is* Nikhilesh, so a transcript labelled "AI:" works against it.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

from ..config import AppConfig

NO_HISTORY = "No previous conversation."


def _label(message: BaseMessage, cfg: AppConfig) -> str:
    role = getattr(message, "type", "")
    return cfg.memory.user_label if role == "human" else cfg.memory.assistant_label


def transcript(messages: Sequence[BaseMessage], cfg: AppConfig) -> str:
    """Plain `Speaker: text` lines."""
    return "\n".join(f"{_label(m, cfg)}: {m.content}" for m in messages)


def for_prompt(messages: Sequence[BaseMessage], summary: str, cfg: AppConfig) -> str:
    """History block for the system prompt's `{chat_history}`."""
    parts: list[str] = []

    if summary:
        parts.append(f"[Earlier conversation summary]\n{summary}")

    if messages:
        parts.append("[Recent conversation]\n" + transcript(messages, cfg))
    elif summary:
        parts.append("[Current conversation]\nNo messages yet.")
    else:
        return NO_HISTORY

    return "\n\n".join(parts)


def for_summary(messages: Sequence[BaseMessage], cfg: AppConfig) -> str:
    """Transcript of the turns being condensed."""
    return transcript(messages, cfg)


__all__ = ["for_prompt", "for_summary", "transcript", "NO_HISTORY"]
