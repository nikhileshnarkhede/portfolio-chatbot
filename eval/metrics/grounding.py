"""
grounding.py - deterministic fabrication detection.

The highest-value metric in the suite, and it needs no judge.

The URL guard catches invented *links*. Nothing catches an invented *employer*,
*date*, or *number* - and that is the worse failure, because a broken link
announces itself while "3 years at Bajaj" does not. This module extracts the
checkable facts from an answer and verifies each against the retrieved context.

Precision matters more than recall here. A metric that cries wolf gets ignored
within a week, so extraction is deliberately conservative:

* **Multi-word capitalised sequences** - high precision for real proper nouns
  ("Bajaj Auto Ltd", "University of Massachusetts Dartmouth").
* **Acronyms** of 2-6 uppercase characters ("PINN", "IGBT", "MLOps").
* **Numbers** of two or more digits, plus percentages and decimals. Bare single
  digits are skipped: they appear in list markers and add noise without
  catching anything.
* **Single capitalised words only when they are not sentence-initial.** "Machine"
  starting a sentence is grammar; "Machine" mid-sentence is probably a name.

Two structural safeguards keep the false-positive rate low. Entities are matched
case-insensitively, so capitalisation drift is not a fabrication. And the
comparison target is the *retrieved context*, not the whole resume - the
question is "did the model use what it was given", which is exactly the claim
groundedness makes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

NAN = float("nan")

# A capitalised token, allowing internal punctuation found in real names:
# "E&ICT", "Tech-Neo", "gpt-oss". NOTE: no "." in the character class - allowing
# it let a match run straight through a full stop and swallow the first word of
# the next sentence ("Machine Learning Researcher. Before").
_CAP_TOKEN = r"[A-Z][A-Za-z0-9&\-']*"

# Connectors are limited to "of/for/de". "and" was tried and removed: it fused
# two unrelated names into one entity ("SOC2 and HIPAA"), which then failed the
# context check as a single phantom fact.
MULTIWORD_RE = re.compile(rf"\b{_CAP_TOKEN}(?:\s+(?:of|for|de)\s+{_CAP_TOKEN}|\s+{_CAP_TOKEN})+\b")
SINGLE_CAP_RE = re.compile(rf"\b{_CAP_TOKEN}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")

# Two number passes. Standalone numbers need 2+ digits to avoid list markers,
# but a SINGLE digit followed by a unit is exactly the fabrication worth
# catching - "7 years of Python", "3 papers".
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
UNIT_NUMBER_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)*)\s*(?:\+\s*)?"
    r"(?=years?|yrs?|months?|weeks?|days?|projects?|models?|papers?|publications?|"
    r"certifications?|internships?|roles?|languages?|people|students?|%)",
    re.IGNORECASE,
)
SENTENCE_START_RE = re.compile(r"(?:^|[.!?:]\s+|\n\s*[-*]?\s*)$")

#: Words that are capitalised for grammar, not because they name anything.
#: Kept short on purpose - the context comparison suppresses most noise on its
#: own, since ordinary English words appear somewhere in a resume.
STOPWORDS = {
    "i", "i'm", "i've", "i'd", "i'll", "my", "me", "the", "a", "an", "and", "but",
    "so", "if", "then", "that", "this", "these", "those", "there", "here", "it",
    "its", "we", "you", "your", "they", "he", "she", "his", "her", "what", "when",
    "where", "which", "who", "why", "how", "yes", "no", "not", "great", "thanks",
    "thank", "sure", "happy", "feel", "free", "let", "would", "could", "should",
    "absolutely", "definitely", "honestly", "basically", "overall", "however",
    "meanwhile", "additionally", "beyond", "across", "during", "while", "since",
}

#: Extracted but never counted as ungrounded: these are properties of the
#: conversation, not claims about the resume.
IGNORE_ENTITIES = {"nikhilesh", "narkhede", "nikhilesh narkhede"}


@dataclass(frozen=True)
class Entity:
    text: str
    kind: str  # "name" | "acronym" | "number"

    @property
    def key(self) -> str:
        return self.text.strip().lower().rstrip(".,;:")


def _is_sentence_initial(text: str, start: int) -> bool:
    return bool(SENTENCE_START_RE.search(text[:start]))


def _strip_leading_stopwords(phrase: str) -> str:
    """Drop grammar words from the front of a multi-word match.

    "My PINN" is not an entity; "PINN" is. Without this, a possessive pronoun
    turns a real, grounded acronym into a phantom phrase that no context can
    support.
    """
    words = phrase.split()
    while words and words[0].lower().rstrip(".,;:") in STOPWORDS:
        words.pop(0)
    return " ".join(words)


def extract_entities(text: str) -> list[Entity]:
    """Checkable facts in an answer: names, acronyms, numbers."""
    if not text:
        return []

    found: dict[str, Entity] = {}
    claimed: list[tuple[int, int]] = []

    def add(entity: Entity) -> None:
        text_ = _strip_leading_stopwords(entity.text)
        if not text_:
            return
        cleaned = Entity(text_, entity.kind)
        if cleaned.key and cleaned.key not in STOPWORDS and cleaned.key not in IGNORE_ENTITIES:
            found.setdefault(cleaned.key, cleaned)

    # Multi-word names first, and remember their spans so their component words
    # are not re-extracted as weaker single-token entities.
    for m in MULTIWORD_RE.finditer(text):
        add(Entity(m.group(0), "name"))
        claimed.append(m.span())

    def inside_claimed(span: tuple[int, int]) -> bool:
        return any(a <= span[0] and span[1] <= b for a, b in claimed)

    for m in ACRONYM_RE.finditer(text):
        if not inside_claimed(m.span()):
            add(Entity(m.group(0), "acronym"))

    for m in SINGLE_CAP_RE.finditer(text):
        if inside_claimed(m.span()):
            continue
        if _is_sentence_initial(text, m.start()):
            continue  # capitalised by grammar, not because it names anything
        add(Entity(m.group(0), "name"))

    for m in NUMBER_RE.finditer(text):
        token = m.group(0)
        if len(token.rstrip("%")) < 2:
            continue  # bare single digits are list markers more often than facts
        add(Entity(token, "number"))

    for m in UNIT_NUMBER_RE.finditer(text):
        add(Entity(m.group(1), "number"))

    return list(found.values())


def _present_in(entity: Entity, haystack: str) -> bool:
    """Is this entity supported by the text?

    Numbers must match as whole tokens - "3" appearing inside "2023" is not
    support for a claim of three years. Names match as substrings so that
    "Bajaj Auto" is supported by "Bajaj Auto Ltd".
    """
    if entity.kind == "number":
        return re.search(rf"(?<!\d){re.escape(entity.key.rstrip('%'))}(?!\d)", haystack) is not None
    return entity.key in haystack


@dataclass
class GroundingScores:
    entity_grounding: float = NAN
    n_entities: int = 0
    ungrounded: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    fabricated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def score(turn: dict, case: dict | None = None) -> GroundingScores:
    """Fraction of an answer's checkable entities that the context supports."""
    answer = turn.get("answer") or ""
    context = (turn.get("context") or "").lower()

    entities = extract_entities(answer)
    if not entities:
        # No checkable claims - a refusal, or pure conversational text. Vacuously
        # grounded, and a real outcome rather than a missing measurement.
        return GroundingScores(entity_grounding=1.0, n_entities=0)

    if not context:
        return GroundingScores(entity_grounding=NAN, n_entities=len(entities))

    ungrounded = [e for e in entities if not _present_in(e, context)]
    return GroundingScores(
        entity_grounding=1.0 - len(ungrounded) / len(entities),
        n_entities=len(entities),
        ungrounded=[e.text for e in ungrounded],
        ungrounded_numbers=[e.text for e in ungrounded if e.kind == "number"],
        fabricated=bool(ungrounded),
    )


def entity_set(text: str) -> set[str]:
    """Normalised entity keys, for consistency comparison."""
    return {e.key for e in extract_entities(text)}


def consistency(answers: Iterable[str]) -> float:
    """Mean pairwise Jaccard similarity of entity sets across repeated answers.

    Catches a config that is correct *on average* but unstable per answer. A
    bot that names a different employer on the third asking is disqualifying in
    a way a merely mediocre answer is not.
    """
    sets = [entity_set(a) for a in answers if a and a.strip()]
    if len(sets) < 2:
        return NAN

    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            scores.append(1.0 if not union else len(sets[i] & sets[j]) / len(union))
    return sum(scores) / len(scores) if scores else NAN


__all__ = [
    "score", "GroundingScores", "Entity", "extract_entities",
    "entity_set", "consistency", "STOPWORDS",
]
