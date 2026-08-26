"""
embedder.py - the embedding model, built from config.

Deliberately tiny, and deliberately the *only* place an embedder is
constructed. The original code built one in `ingest.py` and another in
`app.py`, with a comment warning that their kwargs had to match. They did -
but nothing enforced it, and a mismatch (notably `normalize_embeddings`) makes
stored and query vectors incomparable and silently wrecks retrieval ranking
without raising anything.

One factory, one config block, one source of truth. Ingest time and query time
call the same function.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from ..config import AppConfig


def _spec(cfg: AppConfig) -> tuple:
    """Hashable embedding identity, so the cache keys on settings not object id."""
    e = cfg.embedding
    return (e.provider, e.model_name, e.device, e.normalize_embeddings, e.batch_size)


@lru_cache(maxsize=4)
def _build(provider: str, model_name: str, device: str,
           normalize: bool, batch_size: int) -> Embeddings:
    if provider != "huggingface":  # pragma: no cover - config validation catches this
        raise ValueError(f"Unsupported embedding provider: {provider!r}")

    from ..observability import perf

    # Imported lazily: this pulls torch, which costs seconds of import time.
    # Nothing that merely parses or chunks should pay that. The import itself is
    # inside the timer because on a cold container it dominates.
    with perf.timed("embedder_init"):
        from langchain_huggingface import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize, "batch_size": batch_size},
        )
    return embedder


def build_embedder(cfg: AppConfig) -> Embeddings:
    """Return the embedder for this config. Cached per distinct setting tuple."""
    return _build(*_spec(cfg))


__all__ = ["build_embedder"]
