"""
tracing.py - optional LangSmith tracing.

Off by default. `runs/` already captures everything the eval layer needs, and
it captures it locally: a resume contains a phone number and an email address,
and tracing ships every prompt and completion to a third party.

Turn it on when you want the trace UI for debugging a graph run - LangGraph
traces render as a node timeline, which is genuinely useful for working out
where a slow turn went. Set `observability.langsmith.enabled` and export
`LANGSMITH_API_KEY`.

LangSmith is configured entirely through environment variables, which is why
this module sets them rather than passing a client around.
"""

from __future__ import annotations

import os

from ..config import AppConfig

API_KEY_ENV = "LANGSMITH_API_KEY"


class TracingUnavailable(RuntimeError):
    """Tracing was requested but cannot be enabled."""


def enable(cfg: AppConfig, *, strict: bool = False) -> bool:
    """Turn on LangSmith tracing if the config asks for it. Returns whether it is on.

    `strict=False` (the default) degrades quietly: a missing key disables
    tracing rather than failing the run, because losing observability is not a
    reason to lose the answers. Pass `strict=True` when you are debugging
    tracing itself and silence would be confusing.
    """
    settings = cfg.observability.langsmith

    if not settings.enabled:
        os.environ.pop("LANGSMITH_TRACING", None)
        return False

    if not os.environ.get(API_KEY_ENV, "").strip():
        if strict:
            raise TracingUnavailable(
                f"observability.langsmith.enabled is true but {API_KEY_ENV} is not set."
            )
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.project
    return True


def is_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def disable() -> None:
    os.environ.pop("LANGSMITH_TRACING", None)


__all__ = ["enable", "disable", "is_enabled", "TracingUnavailable", "API_KEY_ENV"]
