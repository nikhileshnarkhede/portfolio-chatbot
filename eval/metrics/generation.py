"""
generation.py - was the answer any good?

Split deliberately into two tiers, because they have very different costs and
very different trustworthiness.

**Tier 1 - deterministic.** Fact coverage, persona checks, length, whether the
answer stayed inside the retrieved context. Free, instant, zero variance. Most
of what you want to know about a prompt change lives here, and it runs without
an API key.

**Tier 2 - LLM judge.** Faithfulness and answer relevancy need a model. The
`Judge` protocol below is the seam: pass one in and the scores appear, pass
nothing and they come back as NaN with the rest of the report intact. That is
what keeps the metric-library decision open - a RAGAS, DeepEval or plain-Groq
judge all satisfy this interface, and none of them is a dependency of the
package.

Treat tier-2 numbers with more suspicion than tier-1. An LLM judge on a free
tier, scoring its own family of models, is a noisy instrument. Use it for
direction, not for a two-decimal-place verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

WORD_RE = re.compile(r"[a-z0-9]+")

#: First-person markers. The persona prompt insists the bot IS Nikhilesh, so an
#: answer drifting into "Nikhilesh has experience in..." is a prompt regression
#: that no faithfulness metric would flag.
FIRST_PERSON = (" i ", "i'm ", "i've ", " my ", " me ", "i ")
THIRD_PERSON = ("nikhilesh has", "nikhilesh is", "nikhilesh's experience",
                "he has", "he is", "his experience")


class Judge(Protocol):
    """An LLM-backed scorer. Return a float in [0, 1], or NaN if unavailable."""

    def faithfulness(self, question: str, context: str, answer: str) -> float: ...
    def relevancy(self, question: str, answer: str) -> float: ...


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall((text or "").lower()))


def fact_coverage(answer: str, facts: list[str]) -> float:
    """Fraction of required facts present in the answer, as substrings."""
    if not facts:
        return float("nan")
    lowered = (answer or "").lower()
    return sum(1 for f in facts if f.lower() in lowered) / len(facts)


def context_overlap(answer: str, context: str) -> float:
    """Fraction of the answer's content words that also appear in the context.

    A cheap, judge-free groundedness proxy. It is genuinely rough - shared
    stopwords and common vocabulary inflate it, and a fluent lie built entirely
    from context words scores well. Use it to catch large drops, not to certify
    a specific answer.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return float("nan")
    return len(answer_tokens & _tokens(context)) / len(answer_tokens)


def first_person(answer: str) -> float:
    """1.0 if the answer speaks as Nikhilesh, 0.0 if it slips into third person."""
    lowered = f" {(answer or '').lower()} "
    if any(marker in lowered for marker in THIRD_PERSON):
        return 0.0
    return 1.0 if any(marker in lowered for marker in FIRST_PERSON) else 0.0


@dataclass
class GenerationScores:
    fact_coverage: float = float("nan")
    missing_facts: list[str] = field(default_factory=list)
    context_overlap: float = float("nan")
    first_person: float = float("nan")
    answer_chars: int = 0
    empty: bool = False
    faithfulness: float = float("nan")
    relevancy: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def score(turn: dict, case: dict, judge: Judge | None = None) -> GenerationScores:
    answer = turn.get("answer") or ""
    context = turn.get("context") or ""
    facts = case.get("must_contain") or []

    scores = GenerationScores(
        fact_coverage=fact_coverage(answer, facts),
        missing_facts=[f for f in facts if f.lower() not in answer.lower()],
        context_overlap=context_overlap(answer, context) if context else float("nan"),
        first_person=first_person(answer),
        answer_chars=len(answer),
        empty=not answer.strip(),
    )

    if judge is not None:
        question = turn.get("question", "")
        try:
            scores.faithfulness = judge.faithfulness(question, context, answer)
            scores.relevancy = judge.relevancy(question, answer)
        except Exception:
            # A judge failure must not sink a run whose tier-1 scores are fine.
            pass

    return scores


__all__ = [
    "score", "GenerationScores", "Judge", "fact_coverage",
    "context_overlap", "first_person",
]
