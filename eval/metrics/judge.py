"""
judge.py - an LLM scorer for faithfulness and relevancy.

Satisfies the `Judge` protocol in `generation.py`.

**Read this before trusting a number from here.** A model scoring output from
its own family, on a few dozen cases, is a noisy instrument. Concretely:

* 0.72 -> 0.79 is not a result. It is inside the range two identical runs
  produce.
* 0.85 -> 0.40 is a result, and worth investigating.
* When a judged score and a deterministic one disagree, believe the
  deterministic one. `entity_grounding` measures most of the same thing with
  none of this uncertainty.

Four design choices keep the noise down:

**The strongest available model**, not the generation model. A weak judge
produces numbers that look like measurements and are not. It also draws from a
separate Groq quota, so judging does not consume the budget that produces the
answers.

**`temperature = 0`, non-streaming.** A judge that samples disagrees with
itself between runs, which would show up as system instability that isn't real.

**Claim-level faithfulness**, not a holistic score. The model splits the answer
into claims and marks each supported or not; counting supported claims is far
more stable than asking for a number, and it yields the unsupported claims
themselves - which is what you actually want when a score drops.

**Failure returns NaN, never 0.0.** A judge that could not be reached must not
look like an answer that scored badly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from portfolio_chatbot.config import AppConfig
from portfolio_chatbot.llm.fallback import run_with_fallback
from portfolio_chatbot.llm.provider import build_llm

NAN = float("nan")

FAITHFULNESS_PROMPT = """You are grading whether an answer is supported by its source material.

Break the ANSWER into individual factual claims. For each claim, decide whether
the CONTEXT directly supports it.

Rules:
- Judge support only. Not whether the claim is true in the world, not whether the
  writing is good, not whether it fully answers anything.
- Conversational filler ("happy to tell you more", "great question") is not a
  factual claim. Skip it.
- A claim that generalises beyond the context is NOT supported.
- If the answer makes no factual claims at all, return an empty claims list.

Return ONLY a JSON object, no prose, in exactly this form:
{{"claims": [{{"claim": "...", "supported": true}}, {{"claim": "...", "supported": false}}]}}

CONTEXT:
{context}

ANSWER:
{answer}

JSON:"""

RELEVANCY_PROMPT = """You are grading whether an answer addresses the question asked.

Score 0 to 10:
  10 - fully answers what was asked
   7 - answers it, with noticeable padding or drift
   4 - partially answers, or answers a related but different question
   0 - does not address the question at all

A polite, correct refusal for information that is genuinely unavailable scores 8:
declining is the right behaviour there, not a failure.

Return ONLY a JSON object, no prose: {{"score": <0-10>}}

QUESTION:
{question}

ANSWER:
{answer}

JSON:"""

SUPPORT_PROMPT = """Decide whether the CONTEXT supports the CLAIM.

Judge support only - not whether the claim is true in the world.

Return ONLY: {{"supported": true}} or {{"supported": false}}

CONTEXT:
{context}

CLAIM:
{claim}

JSON:"""


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Models append explanations despite being told not to, and some wrap output
    in a markdown fence. Both are recoverable, and a best-effort parse beats
    discarding a usable score.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass
class GroqJudge:
    """LLM judge backed by the strongest model in the configured chain."""

    cfg: AppConfig
    model: str | None = None
    _cache: dict[str, Any] = field(default_factory=dict)
    calls: int = 0
    failures: int = 0
    unsupported_claims: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = self.model or self.cfg.judge_model
        # A judge config, not the pipeline's: temperature 0 and no streaming,
        # regardless of how the system under test is configured.
        self._judge_cfg = self.cfg.model_copy(update={
            "llm": self.cfg.llm.model_copy(update={
                "temperature": self.cfg.eval.judge.temperature,
                "streaming": False,
                "model_chain": [self.model],
            })
        })

    # ---------------------------------------------------------------- internals

    def _key(self, *parts: str) -> str:
        return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]

    def _ask(self, prompt: str) -> str | None:
        def call(model: str) -> Any:
            return build_llm(self._judge_cfg, model).invoke(prompt)

        self.calls += 1
        try:
            return run_with_fallback(self._judge_cfg, call).text
        except Exception:
            self.failures += 1
            return None

    def _clip(self, context: str) -> str:
        return context[: self.cfg.eval.judge.max_context_chars]

    # ---------------------------------------------------------------- protocol

    def faithfulness(self, question: str, context: str, answer: str) -> float:
        """Fraction of the answer's factual claims that the context supports."""
        if not answer.strip() or not context.strip():
            return NAN

        key = self._key("faith", question, context, answer)
        if key in self._cache:
            return self._cache[key]

        raw = self._ask(FAITHFULNESS_PROMPT.format(
            context=self._clip(context), answer=answer))
        parsed = extract_json(raw or "")
        score = NAN

        if parsed is not None:
            claims = parsed.get("claims")
            if isinstance(claims, list):
                if not claims:
                    # No factual claims made - vacuously faithful. A real
                    # outcome for a refusal, not a missing measurement.
                    score = 1.0
                else:
                    valid = [c for c in claims if isinstance(c, dict)]
                    if valid:
                        supported = [c for c in valid if c.get("supported")]
                        score = len(supported) / len(valid)
                        self.unsupported_claims.extend(
                            str(c.get("claim", ""))[:160]
                            for c in valid if not c.get("supported")
                        )

        self._cache[key] = score
        return score

    def relevancy(self, question: str, answer: str) -> float:
        """0-1: does the answer address the question asked?"""
        if not answer.strip():
            return NAN

        key = self._key("rel", question, answer)
        if key in self._cache:
            return self._cache[key]

        raw = self._ask(RELEVANCY_PROMPT.format(question=question, answer=answer))
        parsed = extract_json(raw or "")
        score = NAN
        if parsed is not None:
            try:
                score = max(0.0, min(10.0, float(parsed.get("score")))) / 10.0
            except (TypeError, ValueError):
                score = NAN

        self._cache[key] = score
        return score

    def supports(self, context: str, claim: str) -> bool | None:
        """Single-claim verdict. Used by the calibration set, where each case is
        already one claim and decomposition would add noise."""
        key = self._key("sup", context, claim)
        if key in self._cache:
            return self._cache[key]

        raw = self._ask(SUPPORT_PROMPT.format(context=self._clip(context), claim=claim))
        parsed = extract_json(raw or "")
        verdict = None
        if parsed is not None and isinstance(parsed.get("supported"), bool):
            verdict = parsed["supported"]

        self._cache[key] = verdict
        return verdict

    def stats(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "cached": len(self._cache),
            "unsupported_claims": self.unsupported_claims[:20],
        }


def build_judge(cfg: AppConfig, model: str | None = None) -> GroqJudge:
    return GroqJudge(cfg=cfg, model=model)


__all__ = [
    "GroqJudge", "build_judge", "extract_json",
    "FAITHFULNESS_PROMPT", "RELEVANCY_PROMPT", "SUPPORT_PROMPT",
]
