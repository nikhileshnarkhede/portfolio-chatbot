"""
fallback.py - the rate-limit fallback chain, isolated and testable.

Groq's free tier gives every model its own RPM/RPD/TPM/TPD quota, so a
429 on the primary is not "we are out of capacity", it is "try the next
model". The original app.py handled this inline inside the Streamlit render
loop, which meant the logic could only be exercised by actually hitting a rate
limit in a browser.

Here it is a pure function over a callable. `run_with_fallback` takes
`fn(model_name) -> Iterable[str] | str`, walks the model chain, and returns
everything the eval layer needs to reconstruct what happened:

* the text
* one `ModelAttempt` per model tried, in order, including the failures
* which model actually produced the answer
* where the next turn should start, so an exhausted primary is not re-hit
  every single turn

Streaming and non-streaming share this path. Pass `on_chunk` to render tokens
live (the UI does); omit it and the chunks are simply accumulated (eval does).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..state import ModelAttempt, TokenUsage

#: Substrings that mark a rate-limit rejection rather than a real failure.
RATE_LIMIT_MARKERS = ("rate_limit", "rate limit", "429", "too many requests")


def is_rate_limited(error: BaseException | str) -> bool:
    msg = str(error).lower()
    return any(m in msg for m in RATE_LIMIT_MARKERS)


class AllModelsFailed(RuntimeError):
    """Every model in the chain failed. Carries the per-model attempts."""

    def __init__(self, attempts: list[ModelAttempt], last_error: BaseException | None):
        self.attempts = attempts
        self.last_error = last_error
        self.rate_limited = bool(last_error) and is_rate_limited(last_error)
        tried = ", ".join(a["model"] for a in attempts) or "none"
        super().__init__(f"All models failed ({tried}). Last error: {last_error}")


def content_of(piece: Any) -> str:
    """Text out of a chunk, a message, or a plain string.

    `str(chunk)` returns the object's REPR, not its text - so a naive fallback
    silently accumulates "content='' additional_kwargs={}..." into the answer.
    Some providers also return content as a list of typed blocks rather than a
    string; both shapes are handled here so callers can hand this function raw
    provider objects and get text back.
    """
    if isinstance(piece, str):
        return piece
    content = getattr(piece, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return "" if content is None else str(content)


def _is_single(produced: Any) -> bool:
    """True for a lone string or message, false for a stream of chunks."""
    return isinstance(produced, str) or hasattr(produced, "content")


def harvest_usage(piece: Any) -> tuple[int, int] | None:
    """Pull (input_tokens, output_tokens) off a message or chunk, if present.

    Providers differ on where usage appears. LangChain normalizes it onto
    `usage_metadata`, but only populates it when the provider actually sends it -
    Groq attaches it to the FINAL streaming chunk, and some versions omit it
    entirely. Returning None (rather than zeros) keeps "not reported" separable
    from "no tokens used".
    """
    usage = getattr(piece, "usage_metadata", None)
    if not usage:
        return None
    try:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass
class FallbackResult:
    text: str
    model_used: str
    attempts: list[ModelAttempt] = field(default_factory=list)
    next_start_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_available: bool = False
    ttft_s: float | None = None

    @property
    def fell_back(self) -> bool:
        return len(self.attempts) > 1

    def usage(self, cfg: AppConfig) -> TokenUsage:
        """Token counts plus cost, priced from `llm.pricing`."""
        rate = cfg.llm.rate_for(self.model_used)
        cost = (self.input_tokens / 1_000_000) * rate[0] + \
               (self.output_tokens / 1_000_000) * rate[1]
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            cost_usd=round(cost, 6),
            available=self.usage_available,
        )


def _attempt_order(cfg: AppConfig, start_index: int) -> list[int]:
    """Model indices to try: start where we last succeeded, then wrap around."""
    n = len(cfg.llm.model_chain)
    start = start_index % n if n else 0
    if not cfg.llm.fallback_on_rate_limit:
        return [start]
    return list(range(start, n)) + list(range(0, start))


def run_with_fallback(
    cfg: AppConfig,
    fn: Callable[[str], Iterable[str] | str],
    *,
    start_index: int = 0,
    on_chunk: Callable[[str], None] | None = None,
) -> FallbackResult:
    """Run `fn` against the model chain until one succeeds.

    A rate-limit error moves to the next model. Any other error aborts
    immediately - retrying a malformed prompt or a bad API key on three more
    models just burns quota and hides the real cause.
    """
    attempts: list[ModelAttempt] = []
    last_error: BaseException | None = None

    for position, index in enumerate(_attempt_order(cfg, start_index)):
        model = cfg.llm.model_chain[index]
        started = time.perf_counter()
        buffer: list[str] = []

        tokens_in = tokens_out = 0
        saw_usage = False
        first_content_at: float | None = None

        try:
            produced = fn(model)
            if _is_single(produced):
                # Non-streaming: nothing is visible until the whole answer
                # arrives, so time-to-first-token IS total latency. Recorded
                # rather than left null, because that equality is the honest
                # answer to "how long before the user sees something".
                first_content_at = time.perf_counter()
                reported = harvest_usage(produced)
                if reported:
                    tokens_in, tokens_out = reported
                    saw_usage = True
                text = content_of(produced)
                buffer.append(text)
                if on_chunk:
                    on_chunk(text)
            else:
                for piece in produced:
                    reported = harvest_usage(piece)
                    if reported:
                        # Usage normally arrives once, on the final chunk. Summing
                        # rather than assigning keeps it correct if a provider
                        # emits it incrementally instead.
                        tokens_in += reported[0]
                        tokens_out += reported[1]
                        saw_usage = True
                    text = content_of(piece)
                    # First NON-EMPTY chunk. Providers commonly emit an empty
                    # role-only chunk first; counting it would flatter TTFT by
                    # measuring a packet the reader cannot see.
                    if first_content_at is None and text.strip():
                        first_content_at = time.perf_counter()
                    buffer.append(text)
                    if on_chunk:
                        on_chunk(text)

            ttft = round(first_content_at - started, 3) if first_content_at else None
            attempts.append(ModelAttempt(
                model=model, ok=True, rate_limited=False, error=None,
                latency_s=round(time.perf_counter() - started, 3),
                input_tokens=tokens_in, output_tokens=tokens_out, ttft_s=ttft,
            ))
            return FallbackResult(
                text="".join(buffer), model_used=model,
                attempts=attempts, next_start_index=index if cfg.llm.remember_working_model else 0,
                input_tokens=tokens_in, output_tokens=tokens_out, usage_available=saw_usage,
                ttft_s=ttft,
            )

        except Exception as exc:  # noqa: BLE001 - deliberately broad; recorded below
            last_error = exc
            limited = is_rate_limited(exc)
            attempts.append(ModelAttempt(
                model=model, ok=False, rate_limited=limited, error=str(exc)[:300],
                latency_s=round(time.perf_counter() - started, 3),
                input_tokens=tokens_in, output_tokens=tokens_out, ttft_s=None,
            ))
            if not limited:
                break
            del position  # only the ordering mattered

    raise AllModelsFailed(attempts, last_error)


__all__ = [
    "run_with_fallback", "FallbackResult", "AllModelsFailed",
    "is_rate_limited", "harvest_usage", "content_of", "RATE_LIMIT_MARKERS",
]
