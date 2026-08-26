"""
link_guard.py - the URL allowlist, as an auditable function.

A port of `sanitize_links` from the original app.py, with one change that
matters for evaluation: it returns *what it removed*, not just the cleaned
text.

The original stripped bad URLs silently. That worked as a safety net but made
the failure invisible - you could never tell whether the prompt was doing its
job or whether the guard was quietly catching a hallucinated link on every
single turn. Those are very different states of the world, and prompt
experiments hinge on the difference.

This is a hard net that fires regardless of what the prompt says. Keep it that
way: a prompt that behaves is cheaper, but a prompt is a request, not a
guarantee.
"""

from __future__ import annotations

import re

from ..config import AppConfig
from ..state import LinkAudit

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# Negative lookbehind for "(" so URLs already inside markdown syntax - which
# the first pass has by then either kept or rewritten - are not matched again.
BARE_URL_RE = re.compile(r"(?<!\()(https?://[^\s\)\"'<>]+)")


def sanitize(text: str, cfg: AppConfig) -> tuple[str, LinkAudit]:
    """Strip every URL not on the allowlist. Returns (clean_text, audit).

    Markdown links keep their label text when the URL is rejected, so the
    sentence still reads properly; bare URLs are removed outright.
    """
    guard = cfg.safety.url_allowlist
    kept: list[str] = []
    stripped: list[str] = []

    if not guard.enabled:
        return text, LinkAudit(kept=[], stripped=[], stripped_count=0)

    allowed = guard.normalized

    def _norm(url: str) -> str:
        url = url.strip().rstrip(")")
        return url.rstrip("/") if guard.ignore_trailing_slash else url

    def _markdown(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        if _norm(url) in allowed:
            kept.append(url.strip())
            return match.group(0)
        stripped.append(url.strip())
        return label if guard.keep_markdown_label else ""

    text = MARKDOWN_LINK_RE.sub(_markdown, text)

    if guard.strip_bare_urls:
        def _bare(match: re.Match) -> str:
            url = match.group(0).strip()
            if _norm(url) in allowed:
                kept.append(url)
                return match.group(0)
            stripped.append(url)
            return ""

        text = BARE_URL_RE.sub(_bare, text)

    # Removing an inline link can leave doubled spaces mid-sentence.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text, LinkAudit(
        kept=_unique(kept),
        stripped=_unique(stripped),
        stripped_count=len(stripped),
    )


def _unique(items: list[str]) -> list[str]:
    """De-duplicate while preserving first-seen order."""
    seen: set[str] = set()
    out = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


__all__ = ["sanitize", "MARKDOWN_LINK_RE", "BARE_URL_RE"]
