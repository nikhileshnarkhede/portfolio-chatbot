"""
chunker.py - ParsedItems -> Documents, under a swappable strategy.

`config.ingestion.strategy` selects the strategy. Two are implemented:

``structural``
    Structure-aware. One chunk per semantic item, with a header-preserving
    safeguard that splits only chunks exceeding `max_chunk_chars`. This is the
    port of the original `ingest.py` and is the baseline arm.

``recursive``
    Deliberately naive. Strips all markup and splits the whole document on
    size alone, exactly as a generic RAG tutorial would. This is the contrast
    arm: it answers "does structure-aware chunking actually earn its
    complexity on this corpus?"

    **Read this before comparing the two.** Naive splitting destroys the
    `chunk_type` metadata that retrieval routing filters on, so a `recursive`
    run also loses type routing and falls back to MMR for every question. The
    comparison is therefore pipeline-vs-pipeline, not chunker-vs-chunker. That
    is a fair and useful experiment, but do not report the delta as though it
    isolated the chunking step alone.

``fixed`` and ``semantic`` are declared in the config's schema but raise
NotImplementedError - better a loud failure than a silent fallback that
produces numbers you cannot attribute.

Chunk ids
---------
Every Document carries a stable `chunk_id`, which the original ingest.py had
no notion of. Retrieval metrics need golden answers to name specific chunks,
and those references have to survive a re-ingest.

The id is a hash of (section, chunk_type, identity, part) where `identity` is
built from the item's *identifying* attributes - name, title, role, degree,
company, org, provider, type - and deliberately excludes volatile ones like
`period` and `year`. Consequences, both intended:

* Fixing a typo inside a chunk keeps its id, so golden references stay valid.
* Renaming a project changes its id, because it is arguably a different chunk.
* Reordering the resume changes nothing.

A collision counter guards the case of two items sharing every identifying
attribute, so ids are unique even then.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import AppConfig
from .parser import HEADER_RE, ParsedItem, parse, strip_markup

#: Attributes that identify an item. Ordered for readable identity strings.
#: `period`, `year` and `date` are excluded on purpose - editing a date should
#: not invalidate a golden-set reference.
IDENTITY_ATTRS: tuple[str, ...] = (
    "name", "title", "role", "degree", "company", "org", "provider", "venue", "type",
)


def _identity(item: ParsedItem) -> str:
    """A stable, human-readable identity for an item."""
    parts = [item.attrs[a] for a in IDENTITY_ATTRS if item.attrs.get(a)]
    if parts:
        return " | ".join(parts)
    # Leaf sections and intros have no identifying attributes; their
    # section+chunk_type pair is already unique.
    return item.header or item.chunk_type


def _chunk_id(section: str, chunk_type: str, identity: str, part: int, nonce: int = 0) -> str:
    payload = f"{section}|{chunk_type}|{identity}|{part}"
    if nonce:
        payload = f"{payload}|{nonce}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _assign_ids(docs: list[Document]) -> list[Document]:
    """Stamp unique `chunk_id`s, disambiguating any collision deterministically."""
    seen: set[str] = set()
    for doc in docs:
        meta = doc.metadata
        nonce = 0
        while True:
            cid = _chunk_id(
                meta.get("section", ""), meta.get("chunk_type", ""),
                meta.get("identity", ""), meta.get("part", 0), nonce,
            )
            if cid not in seen:
                break
            nonce += 1
        seen.add(cid)
        meta["chunk_id"] = cid
    return docs


def _base_metadata(item: ParsedItem, cfg: AppConfig) -> dict:
    """Tag attributes first; our own keys always win."""
    meta: dict = dict(item.attrs)
    meta.update({
        "source": cfg.paths.raw_resume.rsplit("/", 1)[-1],
        "section": item.section,
        "chunk_type": item.chunk_type,
        "kind": item.kind,
        "header": item.header,
        "identity": _identity(item),
        "part": 0,
        "n_parts": 1,
    })
    return meta


def _header_aware_split(doc: Document, cfg: AppConfig) -> list[Document]:
    """Pass short docs through untouched; split long ones without orphaning them.

    A long chunk is split only after peeling off its leading Markdown header
    block, which is then re-attached to every piece. Without that, piece 2 of a
    project description arrives at the LLM with no indication of which project
    it describes.
    """
    split = cfg.ingestion.split
    if not split.enabled or len(doc.page_content) <= split.max_chunk_chars:
        return [doc]

    lines = doc.page_content.split("\n")
    h = 0
    while h < len(lines) and (lines[h].lstrip().startswith("#") or lines[h].strip() == ""):
        h += 1
    header_block = "\n".join(lines[:h]).strip() if split.reattach_header else ""
    body = "\n".join(lines[h:]).strip() if split.reattach_header else doc.page_content

    if not body:  # nothing but headings
        return [doc]

    inner = max(split.min_inner_chunk_chars, split.max_chunk_chars - len(header_block) - 1)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=inner,
        chunk_overlap=split.chunk_overlap,
        separators=list(split.separators),
    )
    pieces = splitter.split_text(body)

    out = []
    for i, piece in enumerate(pieces):
        content = f"{header_block}\n{piece}".strip() if header_block else piece
        meta = dict(doc.metadata)
        meta["part"] = i
        meta["n_parts"] = len(pieces)
        out.append(Document(page_content=content, metadata=meta))
    return out


# ==========================================================================
# Strategies
# ==========================================================================

def chunk_structural(raw: str, cfg: AppConfig) -> list[Document]:
    """One chunk per semantic item, oversized ones split header-aware."""
    docs: list[Document] = []
    for item in parse(raw, cfg):
        base = Document(page_content=item.content, metadata=_base_metadata(item, cfg))
        docs.extend(_header_aware_split(base, cfg))
    return _assign_ids(docs)


def chunk_recursive(raw: str, cfg: AppConfig) -> list[Document]:
    """Naive size-based splitting of the whole document. No structure, no types.

    `chunk_type` is set to the literal "text" so nothing downstream crashes on
    a missing key - but no configured route will ever match it, which is the
    documented consequence described in this module's docstring.
    """
    split = cfg.ingestion.split
    text = strip_markup(raw, cfg)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split.max_chunk_chars,
        chunk_overlap=split.chunk_overlap,
        separators=list(split.separators),
    )
    pieces = splitter.split_text(text)

    docs = []
    for i, piece in enumerate(pieces):
        m = HEADER_RE.search(piece)
        docs.append(Document(page_content=piece, metadata={
            "source": cfg.paths.raw_resume.rsplit("/", 1)[-1],
            "section": "",
            "chunk_type": "text",
            "kind": "recursive",
            "header": m.group(1).strip() if m else "",
            "identity": f"recursive:{i}",
            "part": i,
            "n_parts": len(pieces),
        }))
    return _assign_ids(docs)


def _not_implemented(name: str) -> Callable[[str, AppConfig], list[Document]]:
    def _raise(raw: str, cfg: AppConfig) -> list[Document]:
        raise NotImplementedError(
            f"Chunking strategy '{name}' is declared in the config schema but not "
            f"implemented. Implement it in ingestion/chunker.py, or set "
            f"ingestion.strategy to one of: {', '.join(sorted(IMPLEMENTED))}."
        )
    return _raise


STRATEGIES: dict[str, Callable[[str, AppConfig], list[Document]]] = {
    "structural": chunk_structural,
    "recursive": chunk_recursive,
    "fixed": _not_implemented("fixed"),
    "semantic": _not_implemented("semantic"),
}

IMPLEMENTED = {"structural", "recursive"}


def chunk(raw: str, cfg: AppConfig) -> list[Document]:
    """Chunk `raw` using the strategy named in the config."""
    strategy = STRATEGIES.get(cfg.ingestion.strategy)
    if strategy is None:  # pragma: no cover - config validation catches this first
        raise ValueError(f"Unknown chunking strategy: {cfg.ingestion.strategy!r}")
    return strategy(raw, cfg)


def summarize(docs: list[Document]) -> dict[str, int]:
    """Chunk counts by type, for the ingest report and the index manifest."""
    counts: dict[str, int] = {}
    for d in docs:
        t = d.metadata.get("chunk_type", "?")
        counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "chunk", "chunk_structural", "chunk_recursive",
    "STRATEGIES", "IMPLEMENTED", "summarize", "IDENTITY_ATTRS",
]
