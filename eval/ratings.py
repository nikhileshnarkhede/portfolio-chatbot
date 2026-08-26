"""
ratings.py - the human rating store (spec §05).

Dimension 3 is the only one no metric captures: is this answer actually useful
to a recruiter. You score it 0-10 against the rubric, and this persists it.

The design decision that makes a 20-question pass sustainable is the key:
ratings are stored against **a hash of the answer text**, not against the case
or the config. An unchanged answer keeps its rating forever, so a second pass
only asks about answers that actually changed - often a handful rather than
twenty. A rating scheme that made you re-score everything after every tweak
would be abandoned within two sessions, and D3 would quietly become unmeasured.

Two stores, deliberately separate:

* **reference** - the fixed set, scored blind in the dashboard. Counts toward
  the gate.
* **field** - the inline 0-10 control under a chatbot answer. Logged only. You
  can see which config produced it, so it is not blind and must not gate.
  Its value is as a source of new reference questions: a field rating of 3 is a
  case worth adding.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from portfolio_chatbot.config import PROJECT_ROOT

RATINGS_DIR = PROJECT_ROOT / "eval" / "ratings"
FIELD_RATINGS = RATINGS_DIR / "field.json"

MIN_SCORE, MAX_SCORE = 0, 10

#: Anchors from spec §05. Shown beside the input so a 7 in November means the
#: same as a 7 in August.
RUBRIC = {
    (9, 10): "Would send to a recruiter unedited. Accurate, complete, sounds like you.",
    (7, 8): "Good. Minor padding, a missing detail, or slightly off tone. No errors.",
    (5, 6): "Serviceable but thin, generic, or incomplete. No errors.",
    (3, 4): "Weak. Misses the question, over-refuses, or reads as a chatbot.",
    (1, 2): "Contains a factual error, a fabricated detail, or a broken claim.",
    (0, 0): "Actively harmful to your candidacy.",
}


def rubric_for(score: int) -> str:
    for (low, high), text in RUBRIC.items():
        if low <= score <= high:
            return text
    return ""


def answer_hash(answer: str) -> str:
    return hashlib.sha1((answer or "").strip().encode("utf-8")).hexdigest()[:16]


def _path(run_fingerprint: str) -> Path:
    return RATINGS_DIR / f"{run_fingerprint}.json"


@dataclass
class RatingStats:
    n: int = 0
    mean: float = float("nan")
    min: int | None = None
    max: int | None = None
    below_three: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load(run_fingerprint: str) -> dict[str, dict]:
    """All ratings for a run, keyed by `case_id:answer_hash`."""
    path = _path(run_fingerprint)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("ratings", {})


def save_rating(run_fingerprint: str, case_id: str, answer: str, score: int,
                note: str = "", experiment: str = "") -> str:
    """Record one rating. Returns the key it was stored under."""
    if not MIN_SCORE <= int(score) <= MAX_SCORE:
        raise ValueError(f"Rating must be {MIN_SCORE}-{MAX_SCORE}, got {score}")

    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(run_fingerprint)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ratings = existing.get("ratings", {})

    key = f"{case_id}:{answer_hash(answer)}"
    ratings[key] = {
        "case_id": case_id,
        "answer_hash": answer_hash(answer),
        "score": int(score),
        "note": note,
        "experiment": experiment,
        "rated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps({
        "_meta": {
            "description": "Human ratings, keyed by case_id:answer_hash. Keying on the "
                           "answer means an unchanged answer keeps its rating across "
                           "runs, so a second pass only asks about what actually changed.",
            "rubric": {f"{lo}-{hi}": text for (lo, hi), text in RUBRIC.items()},
        },
        "run_fingerprint": run_fingerprint,
        "ratings": ratings,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return key


def get_rating(run_fingerprint: str, case_id: str, answer: str) -> dict | None:
    return load(run_fingerprint).get(f"{case_id}:{answer_hash(answer)}")


def find_rating_anywhere(case_id: str, answer: str) -> dict | None:
    """A rating for this exact answer from ANY run.

    The answer hash is what identifies a rating, so an answer that survives a
    config change carries its score with it. This is what stops a re-rate pass
    from re-asking the same questions.
    """
    if not RATINGS_DIR.exists():
        return None
    key = f"{case_id}:{answer_hash(answer)}"
    for path in sorted(RATINGS_DIR.glob("*.json")):
        if path.name == FIELD_RATINGS.name:
            continue
        try:
            found = json.loads(path.read_text(encoding="utf-8")).get("ratings", {}).get(key)
        except json.JSONDecodeError:
            continue
        if found:
            return found
    return None


def stats(run_fingerprint: str) -> RatingStats:
    ratings = load(run_fingerprint)
    scores = [r["score"] for r in ratings.values() if isinstance(r.get("score"), int)]
    if not scores:
        return RatingStats()
    return RatingStats(
        n=len(scores),
        mean=sum(scores) / len(scores),
        min=min(scores),
        max=max(scores),
        below_three=[r["case_id"] for r in ratings.values() if r.get("score", 99) < 3],
    )


def summary_fields(run_fingerprint: str) -> dict[str, Any]:
    """The rating figures the gate reads."""
    s = stats(run_fingerprint)
    return {
        "human_rating_mean": s.mean,
        "human_rating_min": s.min,
        "human_rating_n": s.n,
        "human_rating_below_three": s.below_three,
    }


def unrated(turns: list[dict], run_fingerprint: str) -> list[dict]:
    """Turns still needing a score, cheapest-first for the rater."""
    return [
        t for t in turns
        if find_rating_anywhere(t.get("case_id", ""), t.get("answer", "")) is None
    ]


# ---------------------------------------------------------------- field ratings

def save_field_rating(question: str, answer: str, score: int,
                      run_fingerprint: str = "", note: str = "") -> None:
    """An inline rating from the chatbot UI. Logged, never gated."""
    if not MIN_SCORE <= int(score) <= MAX_SCORE:
        raise ValueError(f"Rating must be {MIN_SCORE}-{MAX_SCORE}, got {score}")

    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    existing = (json.loads(FIELD_RATINGS.read_text(encoding="utf-8"))
                if FIELD_RATINGS.exists() else {"ratings": []})
    existing.setdefault("_meta", {
        "description": "Inline ratings from the chatbot UI. NOT blind - you can see "
                       "which config produced the answer - so these never count "
                       "toward a gate. Their value is as candidates for the "
                       "reference set: a field rating of 3 is a case worth adding.",
    })
    existing["ratings"].append({
        "question": question, "answer_hash": answer_hash(answer),
        "answer_preview": (answer or "")[:200],
        "score": int(score), "note": note,
        "run_fingerprint": run_fingerprint,
        "rated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    FIELD_RATINGS.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def load_field_ratings() -> list[dict]:
    if not FIELD_RATINGS.exists():
        return []
    return json.loads(FIELD_RATINGS.read_text(encoding="utf-8")).get("ratings", [])


def field_candidates(max_score: int = 5) -> list[dict]:
    """Low-scored field ratings - the questions your reference set is missing."""
    return sorted(
        (r for r in load_field_ratings() if r.get("score", 99) <= max_score),
        key=lambda r: r.get("score", 99),
    )


__all__ = [
    "save_rating", "get_rating", "find_rating_anywhere", "load", "stats",
    "summary_fields", "unrated", "answer_hash", "rubric_for", "RatingStats",
    "save_field_rating", "load_field_ratings", "field_candidates",
    "RUBRIC", "RATINGS_DIR", "MIN_SCORE", "MAX_SCORE",
]
