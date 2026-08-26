"""
retriever_tool.py - vector search, config-driven.

Loads the FAISS index for the config's `index_fingerprint` and runs the search
the config asks for. Two modes, matching the original app.py:

``filtered``  metadata-filtered similarity search over specific chunk_types,
              used when routing matched a category. Pulls a wide `fetch_k` and
              narrows to `k`, so a category question gets complete coverage.
``mmr``       maximal marginal relevance over everything, used for open-ended
              questions or when the filter returned nothing. Trades some
              relevance for diversity via `lambda_mult`.

The mode actually used is returned, not inferred, because it is a field the
eval layer reports on: a routing table change that quietly pushes more
questions onto the MMR path is exactly the kind of regression that is
invisible in the answers themselves.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from ..config import AppConfig
from ..ingestion.embedder import build_embedder

RetrievalMode = str  # "filtered" | "mmr" | "none"

#: Loaded stores, keyed by index fingerprint. An explicit dict rather than
#: lru_cache: the loader needs the AppConfig to build a matching embedder, and
#: threading that through a cache key would mean either hashing the whole
#: config or smuggling it in via module state. The fingerprint already
#: identifies the index and its embedding settings exactly.
_STORES: dict[str, Any] = {}


def load_store(cfg: AppConfig):
    """Load (and cache) the vector store for this config."""
    cached = _STORES.get(cfg.index_fingerprint)
    if cached is not None:
        return cached

    if not cfg.index_path.exists():
        raise FileNotFoundError(
            f"No index at {cfg.index_path}. Build it first:\n"
            f"    python scripts/ingest.py"
            + (f" --experiment {cfg.run.experiment_name}"
               if cfg.run.experiment_name != "default" else "")
        )

    from langchain_community.vectorstores import FAISS

    from ..observability import perf

    with perf.timed("index_load"):
        store = FAISS.load_local(
            str(cfg.index_path),
            build_embedder(cfg),
            allow_dangerous_deserialization=cfg.vectorstore.allow_dangerous_deserialization,
        )
    _STORES[cfg.index_fingerprint] = store
    return store


def search(cfg: AppConfig, search_text: str,
           chunk_types: list[str] | None) -> tuple[list[Document], RetrievalMode]:
    """Retrieve for `search_text`, filtered by `chunk_types` when supplied.

    Falls back to MMR when the filter matches nothing - which happens whenever
    routing picks a category the index has no chunks for, and always under the
    `recursive` chunking strategy, whose chunks carry no types at all.
    """
    store = load_store(cfg)
    r = cfg.retrieval

    if chunk_types:
        try:
            docs = store.similarity_search(
                search_text, k=r.filtered.k, fetch_k=r.filtered.fetch_k,
                filter={"chunk_type": chunk_types},
            )
            if docs:
                return docs, "filtered"
        except Exception:
            # Older/newer FAISS wrappers differ on filter semantics. A filter
            # that cannot be applied must degrade to MMR, never to no results.
            pass

    docs = store.max_marginal_relevance_search(
        search_text, k=r.mmr.k, fetch_k=r.mmr.fetch_k, lambda_mult=r.mmr.lambda_mult,
    )
    return docs, "mmr"


def format_context(docs: list[Document], cfg: AppConfig) -> tuple[str, bool]:
    """Assemble retrieved chunks into the prompt's context block.

    Returns (context, truncated). `truncated` is surfaced in the state because
    a context silently clipped at `max_context_chars` drops whole chunks the
    retriever thought were relevant - a plausible cause of a bad answer that is
    otherwise invisible.
    """
    c = cfg.retrieval.context
    if not docs:
        return c.empty_message, False

    parts = [
        c.section_template.format(i=i, content=d.page_content.strip())
        for i, d in enumerate(docs, 1)
    ]
    context = c.joiner.join(parts)

    if len(context) > c.max_context_chars:
        return context[: c.max_context_chars], True
    return context, False


def register_store(cfg: AppConfig, store: Any) -> None:
    """Inject a store directly. Used by tests to avoid loading a real index."""
    _STORES[cfg.index_fingerprint] = store


def clear_cache() -> None:
    """Drop cached stores. Needed after a re-ingest in a live session."""
    _STORES.clear()


__all__ = ["load_store", "search", "format_context", "register_store", "clear_cache"]
