"""
registry.py - reserved for LLM tool-calling.

Deliberately empty of behaviour.

`tools/` currently holds two things the pipeline calls directly:
`retriever_tool.py` (vector search) and `link_guard.py` (the URL allowlist).
Neither is exposed to the model as a callable tool, because nothing in this
pipeline needs the model to decide when to retrieve - routing and retrieval are
deterministic, which is precisely what makes them measurable.

This file is where a bound tool list would go if that changes, e.g. an agentic
variant that lets the model issue its own follow-up searches. That would be a
genuinely interesting experiment, and a fair bit less predictable: an agent
that chooses its own retrieval makes `route_correct` and `hit@k` much harder to
interpret, since there is no longer one retrieval step to score.

The file is kept rather than deleted so the structure documented in
STRUCTURE.md stays honest about where that code belongs.
"""

from __future__ import annotations

TOOLS: list = []

__all__ = ["TOOLS"]
