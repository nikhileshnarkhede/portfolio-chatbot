"""
retrieval.py - did the right chunks come back, in a good order?

Every metric here is deterministic. No LLM, no API key, no cost, no variance -
run it a thousand times and get the same number. That matters more than it
sounds: retrieval is the half of a RAG system you can measure exactly, and
measuring it exactly means a chunking or `k` change produces a fact rather than
an impression.

Golden cases name chunks by **identity** ("Chat with YouTube Videos"), not by
chunk_id hash. Identities are readable, survive a re-chunk that splits an item
into more parts, and are what a person can actually write by hand. The
resolution from identity to retrieved chunks happens here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


def _identities(chunks: Iterable[dict]) -> list[str]:
    return [str(c.get("identity") or "") for c in chunks]


def _matches(expected: str, identity: str) -> bool:
    """Case-insensitive substring match.

    Substring rather than equality because a chunk's identity is a join of its
    attributes ("Chat with YouTube Videos | RAG / Conversational AI"), and a
    golden case should be able to name just the project.
    """
    return expected.strip().lower() in identity.lower()


def hit_at_k(chunks: list[dict], expected: list[str], k: int | None = None) -> float:
    """1.0 if any expected chunk appears in the top k, else 0.0."""
    if not expected:
        return float("nan")
    window = _identities(chunks[:k] if k else chunks)
    return 1.0 if any(_matches(e, i) for e in expected for i in window) else 0.0


def recall_at_k(chunks: list[dict], expected: list[str], k: int | None = None) -> float:
    """Fraction of expected chunks that appear in the top k.

    The metric that punishes over-narrow retrieval: a question asking for every
    project should return every project, and hit@k cannot see the difference
    between one and eleven.
    """
    if not expected:
        return float("nan")
    window = _identities(chunks[:k] if k else chunks)
    found = sum(1 for e in expected if any(_matches(e, i) for i in window))
    return found / len(expected)


def mrr(chunks: list[dict], expected: list[str]) -> float:
    """Reciprocal rank of the first expected chunk. Rewards ordering, not just presence."""
    if not expected:
        return float("nan")
    for rank, identity in enumerate(_identities(chunks), start=1):
        if any(_matches(e, identity) for e in expected):
            return 1.0 / rank
    return 0.0


def graded_relevance(case: dict) -> dict[str, float]:
    """Identity -> relevance grade for one case.

    Graded rather than binary because not every expected chunk matters equally.
    For "tell me about the Supply Chain Tracker", that project's chunk is the
    answer; a related project is mild context. Binary relevance scores both the
    same and so cannot see a ranking that puts the wrong one first.

    `primary_identities` are grade 2, `expected_identities` grade 1. A case that
    declares neither grade has all its expected chunks at grade 1, which makes
    nDCG degrade gracefully to the binary case.
    """
    grades: dict[str, float] = {}
    for identity in case.get("expected_identities") or []:
        grades[identity] = 1.0
    for identity in case.get("primary_identities") or []:
        grades[identity] = 2.0
    return grades


def ndcg_at_k(chunks: list[dict], case: dict, k: int | None = None) -> float:
    """Normalised discounted cumulative gain.

    Rewards putting the most relevant chunk first, not merely including it.
    The discount is 1/log2(rank+1), so a primary chunk at rank 1 is worth
    substantially more than the same chunk at rank 8 - which matters here
    because `max_context_chars` truncates from the end, and a late chunk may
    never reach the model at all.
    """
    grades = graded_relevance(case)
    if not grades:
        return float("nan")

    window = _identities(chunks[:k] if k else chunks)

    def grade_of(identity: str) -> float:
        for expected, grade in grades.items():
            if _matches(expected, identity):
                return grade
        return 0.0

    gains = [grade_of(identity) for identity in window]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    ideal = sorted(grades.values(), reverse=True)[: len(window) or None]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else float("nan")


def type_precision(chunks: list[dict], expected_types: list[str]) -> float:
    """Fraction of retrieved chunks whose chunk_type was asked for.

    A proxy for context precision that needs no judge: if a question routed to
    `project` and half the context is `experience`, the prompt is paying tokens
    for text that cannot answer it.
    """
    if not expected_types or not chunks:
        return float("nan")
    wanted = {t.lower() for t in expected_types}
    hits = sum(1 for c in chunks if str(c.get("chunk_type", "")).lower() in wanted)
    return hits / len(chunks)


def route_correct(actual_route: dict | None, expected_rule: str | None) -> float:
    """Did the routing table fire the rule the case expected?

    `expected_rule = None` asserts that nothing should match, which is how you
    pin the open-ended questions that are supposed to reach MMR.
    """
    actual = (actual_route or {}).get("rule_name")
    return 1.0 if actual == expected_rule else 0.0


@dataclass
class RetrievalScores:
    hit_at_k: float = float("nan")
    recall_at_k: float = float("nan")
    mrr: float = float("nan")
    ndcg_at_k: float = float("nan")
    type_precision: float = float("nan")
    route_correct: float = float("nan")
    n_chunks: int = 0
    mode: str = ""
    context_chars: int = 0
    context_truncated: bool = False
    missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def score(turn: dict, case: dict, k: int | None = None) -> RetrievalScores:
    """Score one turn's retrieval against one golden case."""
    chunks = turn.get("chunks") or []
    expected = case.get("expected_identities") or []
    expected_types = case.get("expected_chunk_types") or []

    window = _identities(chunks[:k] if k else chunks)
    missing = [e for e in expected if not any(_matches(e, i) for i in window)]

    return RetrievalScores(
        hit_at_k=hit_at_k(chunks, expected, k),
        recall_at_k=recall_at_k(chunks, expected, k),
        mrr=mrr(chunks, expected),
        ndcg_at_k=ndcg_at_k(chunks, case, k),
        type_precision=type_precision(chunks, expected_types),
        route_correct=route_correct(turn.get("route"), case.get("expected_route")),
        n_chunks=len(chunks),
        mode=turn.get("retrieval_mode", ""),
        context_chars=turn.get("context_chars", 0),
        context_truncated=bool(turn.get("context_truncated")),
        missing=missing,
    )


def chunk_coverage(seen_chunk_ids: set[str], manifest: dict) -> dict:
    """Which indexed chunks were reached by at least one query in the suite.

    A different question from every other retrieval metric. Those ask "did we
    find the right chunk for THIS question"; this asks whether any question
    reaches a given chunk at all. An orphaned chunk - content in the resume that
    no query surfaces - is invisible to question-driven metrics, because no case
    was ever written for the content nobody thought to ask about.
    """
    all_chunks = manifest.get("chunks") or []
    if not all_chunks:
        return {"chunk_coverage": float("nan"), "n_indexed": 0,
                "n_reached": 0, "orphans": []}

    indexed = {c.get("chunk_id"): c for c in all_chunks if c.get("chunk_id")}
    reached = seen_chunk_ids & set(indexed)
    orphans = [
        {"chunk_id": cid, "chunk_type": c.get("chunk_type"), "identity": c.get("identity")}
        for cid, c in indexed.items() if cid not in reached
    ]
    return {
        "chunk_coverage": len(reached) / len(indexed),
        "n_indexed": len(indexed),
        "n_reached": len(reached),
        "orphans": sorted(orphans, key=lambda o: str(o["identity"]))[:20],
    }


__all__ = [
    "score", "RetrievalScores", "hit_at_k", "recall_at_k", "mrr",
    "ndcg_at_k", "graded_relevance", "type_precision", "route_correct",
    "chunk_coverage",
]
