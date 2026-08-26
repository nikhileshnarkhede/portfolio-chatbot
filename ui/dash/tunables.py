"""
tunables.py - which config knobs the dashboard exposes, and how.

A declarative registry rather than reflection over the Pydantic model. Three
reasons: not every config value is worth a slider (the URL allowlist is not a
knob), the sensible *range* for a parameter is not derivable from its type, and
each one needs a sentence explaining what moving it actually does.

`rebuilds_index` is the field that matters most. Changing it invalidates the
vector index, so the dashboard must warn and force a re-ingest before any
result is trustworthy - otherwise you would be scoring a new chunking strategy
against the old index's vectors and reading the numbers as if they meant
something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Knob:
    path: str                 # dotted config path
    label: str
    kind: str                 # "slider" | "int" | "float" | "select" | "toggle"
    help: str
    group: str
    rebuilds_index: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: tuple[Any, ...] | None = None


KNOBS: tuple[Knob, ...] = (
    # ---------------- chunking: every change here rebuilds the index ----------
    Knob("ingestion.strategy", "Chunking strategy", "select", group="Chunking",
         rebuilds_index=True, options=("structural", "recursive"),
         help="'structural' keeps one chunk per resume item. 'recursive' splits on "
              "size alone and destroys chunk_type metadata, so routing cannot fire - "
              "compare it as pipeline-vs-pipeline, not chunker-vs-chunker."),
    Knob("ingestion.split.max_chunk_chars", "Max chunk size", "slider", group="Chunking",
         rebuilds_index=True, minimum=256, maximum=2400, step=64,
         help="Smaller chunks raise context precision but fragment multi-fact items, "
              "so one experience entry may span several chunks and only some get "
              "retrieved. Watch recall_at_k when lowering this."),
    Knob("ingestion.split.chunk_overlap", "Chunk overlap", "slider", group="Chunking",
         rebuilds_index=True, minimum=0, maximum=400, step=16,
         help="Characters repeated between adjacent pieces of a split chunk. Guards "
              "against a fact landing exactly on a boundary."),

    # ---------------- retrieval: no re-ingest needed ------------------------
    Knob("retrieval.filtered.k", "Filtered results (k)", "slider", group="Retrieval",
         minimum=2, maximum=40, step=1,
         help="How many chunks a routed question retrieves. Raising it helps 'list "
              "everything' questions and pushes the context toward its ceiling."),
    Knob("retrieval.filtered.fetch_k", "Filtered candidate pool", "slider", group="Retrieval",
         minimum=10, maximum=200, step=10,
         help="Candidates considered before narrowing to k. Must be >= k."),
    Knob("retrieval.mmr.k", "MMR results (k)", "slider", group="Retrieval",
         minimum=2, maximum=30, step=1,
         help="Chunks retrieved for open-ended questions, where no routing rule fired."),
    Knob("retrieval.mmr.lambda_mult", "MMR diversity", "slider", group="Retrieval",
         minimum=0.0, maximum=1.0, step=0.05,
         help="1.0 is pure relevance; lower trades relevance for diversity. Matters "
              "for 'tell me about yourself', where one topic dominating is a failure."),
    Knob("retrieval.context.max_context_chars", "Context budget", "slider", group="Retrieval",
         minimum=2000, maximum=16000, step=500,
         help="Hard ceiling on the assembled context. When it clips, chunks the "
              "retriever ranked as relevant never reach the model - watch "
              "'context truncated' in the scorecard."),
    Knob("retrieval.routing.enabled", "Type routing", "toggle", group="Retrieval",
         help="Off sends every question down the MMR path. The honest control arm "
              "for 'does the routing table earn its complexity?'"),
    Knob("retrieval.query_expansion.mode", "Query expansion", "select", group="Retrieval",
         options=("keyword_rules", "llm", "none"),
         help="'keyword_rules' appends domain terms for free. 'llm' can resolve "
              "follow-up pronouns, which keyword rules fundamentally cannot, at one "
              "extra call per turn. 'none' is the control."),

    # ---------------- generation --------------------------------------------
    Knob("llm.temperature", "Temperature", "slider", group="Generation",
         minimum=0.0, maximum=1.5, step=0.05,
         help="Higher wanders. Note this directly inflates the noise floor: at 0.0 "
              "repeated runs agree, so a smaller real change becomes detectable."),
    Knob("llm.streaming", "Stream responses", "toggle", group="Generation",
         help="Off makes token usage always available (it rides on the final message) "
              "but makes TTFT equal total latency, since nothing is visible until the "
              "whole answer lands."),
    Knob("prompts.system", "System prompt", "select", group="Generation",
         options=("system.v1_persona", "system.v2_persona"),
         help="v1 is byte-identical to the original deployed prompt. v2 is 62% shorter "
              "with compressed URL rules - watch url_attempt_rate, not just quality."),

    # ---------------- memory -------------------------------------------------
    Knob("memory.summarize_after_n_messages", "Summarize after", "slider", group="Memory",
         minimum=4, maximum=40, step=2,
         help="Message count that triggers condensing older turns."),
    Knob("memory.keep_last_n", "Keep last N messages", "slider", group="Memory",
         minimum=2, maximum=20, step=2,
         help="Turns kept verbatim after summarizing. Must be below the trigger."),
    Knob("memory.enabled", "Conversation memory", "toggle", group="Memory",
         help="Off disables summarization entirely - the strict replay of the "
              "original app's behaviour."),
)

GROUPS = tuple(dict.fromkeys(k.group for k in KNOBS))


def get_value(config: dict, path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_value(config: dict, path: str, value: Any) -> dict:
    """Set a dotted path, returning a modified copy."""
    import copy
    out = copy.deepcopy(config)
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return out


def changed_knobs(current: dict, base: dict) -> list[tuple[Knob, Any, Any]]:
    """(knob, was, now) for every knob whose value differs."""
    out = []
    for knob in KNOBS:
        before, after = get_value(base, knob.path), get_value(current, knob.path)
        if before != after:
            out.append((knob, before, after))
    return out


def requires_reingest(current: dict, base: dict) -> bool:
    return any(knob.rebuilds_index for knob, _, _ in changed_knobs(current, base))


__all__ = ["Knob", "KNOBS", "GROUPS", "get_value", "set_value",
           "changed_knobs", "requires_reingest"]
