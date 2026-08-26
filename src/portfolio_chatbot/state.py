"""
state.py - the contract every node reads and writes.

This is the single most important file for the evaluation goal, because of one
design rule:

    **If a number belongs in an eval report, it must exist as a field here.**

A node may not compute something, use it, and throw it away. Retrieval mode,
which route fired, what the model emitted *before* the link guard touched it,
which models were tried and why they failed - all of it lands in the state, so
`observability/run_logger.py` can serialize a complete record of the turn
without re-running anything or guessing.

Consequences worth understanding before adding a field:

* **`draft_answer` and `answer` are both kept.** `draft_answer` is the raw LLM
  output; `answer` is what survives `nodes/sanitize.py`. Collapsing them into
  one field would make URL hallucination permanently unmeasurable - you cannot
  count links the model tried to invent if you only keep the cleaned text.

* **The AppConfig is NOT in the state.** It travels through LangGraph's
  `RunnableConfig` under `configurable["app_config"]`. State is serialized into
  every checkpoint, and a frozen config carrying a 31-entry URL allowlist would
  be copied into each one. The state carries only the two fingerprints, which
  is all a run record needs to tie a result back to its exact configuration.

* **Most fields are last-write-wins.** Only four accumulate, and each has an
  explicit reducer: `messages`, `model_attempts`, `trace`, `timings`. Anything
  else added later should default to replacement unless there is a reason.
  If you do add an accumulating field, clear it in `new_turn` with `RESET` -
  returning an empty list appends nothing rather than clearing, which lets a
  turn's diagnostics leak into the next one.

Access pattern inside a node:

    def retrieve(state: GraphState, config: RunnableConfig) -> dict:
        cfg = config["configurable"]["app_config"]
        ...
        return {"documents": docs, "retrieval_mode": "filtered"}

Nodes return a **partial** dict, never the whole state. LangGraph merges it
using the reducers declared below.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# ==========================================================================
# Reducers
# ==========================================================================

#: Assign to an accumulating field to clear it.
#:
#: This exists because of a genuine trap. An accumulating reducer is called as
#: `reducer(existing, update)`, so returning `[]` from a node does NOT clear
#: the field - it appends nothing and leaves the previous contents intact.
#: With a checkpointer keeping one thread alive across a conversation, that
#: means `model_attempts` and `trace` keep growing turn after turn, and every
#: per-turn metric silently measures the whole conversation instead. `RESET`
#: is the only way to actually empty one of these fields.
#:
#: It is `None` rather than a custom sentinel object on purpose: LangGraph
#: serializes input writes into the checkpoint with msgpack, and a bespoke
#: sentinel class raises `TypeError: Type is not msgpack serializable`. `None`
#: is unambiguous here because a reducer is only invoked when a node actually
#: returns that key - "field absent" and "field set to None" are distinct.
RESET: Any = None


def append_or_reset(left: list | None, right: Any) -> list:
    """Append to a list; `RESET` (None) empties it.

    Used for `trace` and `model_attempts` - fields where several nodes each
    contribute entries within a turn, but which must start empty on the next
    turn.
    """
    if right is None:
        return []
    return list(left or []) + list(right)


def merge_or_reset(left: dict | None, right: Any) -> dict:
    """Shallow-merge, right wins; `RESET` (None) empties.

    Used for `timings`, where each node contributes its own key and must not
    clobber the others. Plain replacement would leave only the last node's
    timing; plain merging would carry timings across turns.
    """
    if right is None:
        return {}
    return {**(left or {}), **right}


# ==========================================================================
# Structured records
# ==========================================================================

class RouteDecision(TypedDict, total=False):
    """Which routing rule fired, and what it selected.

    Recorded even when nothing matched (`matched=False`) - the share of
    questions that fall through to MMR is itself a metric worth watching when
    the routing table changes.
    """
    matched: bool
    rule_name: str | None          # RouteRule.name from the config
    matched_keyword: str | None    # the specific keyword that fired
    chunk_types: list[str]         # the metadata filter applied


class ModelAttempt(TypedDict, total=False):
    """One entry per model tried, in order, including failures.

    The fallback chain is invisible in the final answer, so without this the
    eval layer cannot distinguish "the primary model answered" from "the
    primary was rate-limited and a weaker backup answered". Those are
    different experimental conditions producing the same-looking output.
    """
    model: str
    ok: bool
    rate_limited: bool
    error: str | None
    latency_s: float | None
    ttft_s: float | None
    input_tokens: int
    output_tokens: int


class TokenUsage(TypedDict, total=False):
    """Tokens consumed by one turn, and what they cost.

    `available` is explicit rather than inferred from zeros, because "the
    provider did not report usage" and "this turn used no tokens" are different
    facts and only one of them is a bug. Groq reports usage on the final
    streaming chunk, but not every provider or version does; with
    `llm.streaming: false` it is always present.

    Cost is computed from `llm.pricing` in the config, which defaults to 0.0 -
    the free tier is genuinely free. Fill in your plan's rates to get real
    numbers.
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    available: bool


class LinkAudit(TypedDict, total=False):
    """What `nodes/sanitize.py` kept and removed.

    `stripped` is the direct input to the url_hallucination_rate metric in
    `eval/metrics/safety.py`. A non-empty list means the prompt failed to
    constrain the model and the hard allowlist had to catch it.
    """
    kept: list[str]
    stripped: list[str]
    stripped_count: int


# ==========================================================================
# State
# ==========================================================================

class InputState(TypedDict):
    """What the caller must supply to invoke the graph."""
    question: str


class OutputState(TypedDict):
    """What a caller gets back if it only cares about the reply."""
    answer: str


class GraphState(TypedDict, total=False):
    """The full internal state. `total=False` because nodes return partials."""

    # ---- identity: stamped once per turn, never mutated ----
    turn_id: str
    experiment_name: str
    run_fingerprint: str      # AppConfig.run_fingerprint - ties a result to its config
    index_fingerprint: str    # AppConfig.index_fingerprint - which index answered

    # ---- conversation ----
    # add_messages appends and handles id-based dedup/updates, so a node can
    # return a single new message rather than the whole history.
    messages: Annotated[list[AnyMessage], add_messages]

    # ---- input ----
    question: str             # raw user text, verbatim - routing reads this

    # ---- retrieval ----
    expanded_question: str    # after nodes/expand_query.py; ranking reads this
    route: RouteDecision
    documents: list[Document]
    retrieval_mode: Literal["filtered", "mmr", "none"]
    context: str              # formatted, truncated text handed to the prompt
    context_chars: int
    context_truncated: bool   # true if max_context_chars clipped the context

    # ---- generation ----
    draft_answer: str         # raw LLM output, BEFORE sanitization
    answer: str               # final text shown to the user, AFTER sanitization
    model_used: str
    model_attempts: Annotated[list[ModelAttempt], append_or_reset]

    # Where the fallback chain should start next turn. Deliberately absent from
    # `new_turn` and from EVAL_FIELDS: it must PERSIST across turns, so that an
    # exhausted primary model is not re-tried on every single message. Resetting
    # it each turn would reintroduce a wasted 429 per turn.
    model_start_index: int

    # ---- cost & perceived speed ----
    token_usage: TokenUsage

    # Seconds until the first visible token. Distinct from timings["generate"],
    # which is the whole call: in a streaming UI this is what "fast" means, and
    # a change that improves total latency while delaying the first token makes
    # the app feel slower, not faster.
    ttft_s: float | None

    # ---- safety ----
    link_audit: LinkAudit

    # ---- memory ----
    summary: str              # rolling summary of turns dropped from history
    summarized_this_turn: bool

    # ---- diagnostics ----
    timings: Annotated[dict[str, float], merge_or_reset]
    trace: Annotated[list[str], append_or_reset]
    error: str | None


# ==========================================================================
# Helpers
# ==========================================================================

def new_turn(
    question: str,
    *,
    experiment_name: str,
    run_fingerprint: str,
    index_fingerprint: str,
    summary: str | None = None,
) -> GraphState:
    """Build the initial state for one turn.

    The three accumulating fields are set to `RESET`, not to `[]` / `{}`. That
    distinction is load-bearing: with a checkpointer keeping the thread alive,
    handing an empty list to an appending reducer appends nothing and leaves
    the previous turn's entries in place. Turn 3 would then report three
    `model_attempts` and three `trace` lines, and every per-turn metric would
    quietly describe the whole conversation. `RESET` is what actually clears
    them. See `append_or_reset` / `merge_or_reset` above.

    `summary` and `messages` are the fields that legitimately persist across
    turns. `messages` is never set here at all, and `summary` is omitted unless
    the caller passes one explicitly - passing `summary=""` as a default would
    overwrite the checkpointed summary with an empty string on every turn,
    silently undoing summarization the moment it happened. Pass a value only
    when driving the graph WITHOUT a checkpointer, where nothing else carries
    it forward.
    """
    state: dict = {
        "turn_id": uuid.uuid4().hex[:12],
        "experiment_name": experiment_name,
        "run_fingerprint": run_fingerprint,
        "index_fingerprint": index_fingerprint,
        "question": question,
        "expanded_question": "",
        "route": RouteDecision(matched=False, rule_name=None, matched_keyword=None, chunk_types=[]),
        "documents": [],
        "retrieval_mode": "none",
        "context": "",
        "context_chars": 0,
        "context_truncated": False,
        "draft_answer": "",
        "answer": "",
        "model_used": "",
        "model_attempts": RESET,
        "token_usage": TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0,
                                  cost_usd=0.0, available=False),
        "ttft_s": None,
        "link_audit": LinkAudit(kept=[], stripped=[], stripped_count=0),
        "summarized_this_turn": False,
        "timings": RESET,
        "trace": RESET,
        "error": None,
    }
    if summary is not None:
        state["summary"] = summary
    return cast(GraphState, state)


def chunk_refs(state: GraphState) -> list[dict[str, Any]]:
    """Serializable identifiers for the retrieved chunks.

    The run log stores these rather than full Documents: retrieval metrics
    (hit@k, MRR, context precision) only need to know *which* chunks came back
    and in what order, and full page content would multiply the log size for
    no analytical gain. Set `observability.log_context` to keep the assembled
    context string alongside them when you need to eyeball it.
    """
    refs = []
    for rank, doc in enumerate(state.get("documents") or []):
        meta = doc.metadata or {}
        refs.append({
            "rank": rank,
            # chunk_id and identity are what the eval layer joins on: golden
            # cases name chunks by identity, and chunk_id ties a result back to
            # a specific entry in the index manifest. Omitting them makes every
            # retrieval metric silently score zero.
            "chunk_id": meta.get("chunk_id"),
            "identity": meta.get("identity"),
            "chunk_type": meta.get("chunk_type"),
            "section": meta.get("section"),
            "header": meta.get("header"),
            "part": meta.get("part"),
            "chars": len(doc.page_content),
        })
    return refs


#: Fields `observability/run_logger.py` persists for every turn. Kept here,
#: next to the definitions, so adding a field to the state and forgetting to
#: log it is a one-line fix in an obvious place rather than a silent gap
#: discovered weeks later when a report turns out to be missing a column.
EVAL_FIELDS: tuple[str, ...] = (
    "turn_id",
    "experiment_name",
    "run_fingerprint",
    "index_fingerprint",
    "question",
    "expanded_question",
    "route",
    "retrieval_mode",
    "context_chars",
    "context_truncated",
    "draft_answer",
    "answer",
    "model_used",
    "model_attempts",
    "token_usage",
    "ttft_s",
    "link_audit",
    "summarized_this_turn",
    "timings",
    "error",
)


__all__ = [
    "GraphState",
    "InputState",
    "OutputState",
    "RouteDecision",
    "ModelAttempt",
    "LinkAudit",
    "TokenUsage",
    "new_turn",
    "chunk_refs",
    "RESET",
    "append_or_reset",
    "merge_or_reset",
    "EVAL_FIELDS",
]
