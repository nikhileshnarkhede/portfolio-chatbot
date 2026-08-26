"""
build_index.py - parse -> chunk -> embed -> FAISS, keyed by config fingerprint.

The index for a config lives at `data/index/<index_fingerprint>/`. That is the
mechanism that makes chunking sweeps safe: two configs that chunk and embed
identically share an index and skip the rebuild, while a config that changes
`max_chunk_chars` gets its own directory and cannot silently answer questions
from another experiment's vectors.

Alongside the FAISS files, each index directory gets a `manifest.json`
recording the fingerprint, the chunk counts by type, the full chunk id list,
and the ingestion-relevant config. `eval/` reads that manifest to resolve the
chunk ids referenced in a golden set, and it is what tells you months later
what a given index actually contains.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config import AppConfig
from .chunker import chunk, summarize
from .embedder import build_embedder

MANIFEST_NAME = "manifest.json"


@dataclass
class BuildReport:
    index_path: str
    index_fingerprint: str
    strategy: str
    n_chunks: int
    by_type: dict[str, int]
    chars_total: int
    chars_mean: int
    chars_max: int
    elapsed_s: float
    skipped: bool = False       # index already existed and was reused
    dry_run: bool = False       # chunked only; no embedding, no index written
    dumped_to: str | None = None
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"strategy      {self.strategy}",
            f"fingerprint   {self.index_fingerprint}",
            f"index         {self.index_path}",
            f"chunks        {self.n_chunks}  (mean {self.chars_mean} chars, max {self.chars_max})",
            "",
            "chunks by type:",
        ]
        for t, n in self.by_type.items():
            lines.append(f"  {t:26s} {n}")
        if self.dumped_to:
            lines += ["", f"chunk dump    {self.dumped_to}"]
        for w in self.warnings:
            lines += ["", f"WARNING: {w}"]
        if self.dry_run:
            status = f"dry run - chunked in {self.elapsed_s:.1f}s, nothing embedded or written"
        elif self.skipped:
            status = "index already built for this fingerprint - reused, nothing rebuilt"
        else:
            status = f"built in {self.elapsed_s:.1f}s"
        lines += ["", status]
        return "\n".join(lines)


def _faiss_distance(cfg: AppConfig):
    from langchain_community.vectorstores.utils import DistanceStrategy
    return getattr(DistanceStrategy, cfg.vectorstore.distance_strategy)


def _dump_chunks(docs: list[Document], cfg: AppConfig) -> str:
    """Write every chunk to JSONL so chunking can be inspected without a rebuild."""
    out_dir = cfg.resolve(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chunks_{cfg.index_fingerprint}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({"metadata": d.metadata, "content": d.page_content},
                                ensure_ascii=False) + "\n")
    return cfg.display_path(path)


def _write_manifest(docs: list[Document], cfg: AppConfig, report: BuildReport) -> None:
    manifest = {
        "index_fingerprint": cfg.index_fingerprint,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": cfg.paths.raw_resume,
        "strategy": cfg.ingestion.strategy,
        "n_chunks": report.n_chunks,
        "by_type": report.by_type,
        "ingestion": cfg.ingestion.model_dump(mode="json"),
        "embedding": cfg.embedding.model_dump(mode="json"),
        "vectorstore": cfg.vectorstore.model_dump(mode="json"),
        "chunks": [
            {
                "chunk_id": d.metadata.get("chunk_id"),
                "chunk_type": d.metadata.get("chunk_type"),
                "section": d.metadata.get("section"),
                "identity": d.metadata.get("identity"),
                "part": d.metadata.get("part"),
                "n_parts": d.metadata.get("n_parts"),
                "chars": len(d.page_content),
            }
            for d in docs
        ],
    }
    (cfg.index_path / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build(cfg: AppConfig, *, force: bool = False, dry_run: bool = False,
          embedder: Embeddings | None = None) -> BuildReport:
    """Build (or reuse) the index for `cfg`.

    `dry_run` parses and chunks but never loads the embedding model or writes
    an index - useful for inspecting how a chunking change lands before paying
    for embedding. `embedder` is injectable so tests can exercise the whole
    path with a deterministic fake instead of downloading a model.
    """
    started = time.perf_counter()

    resume = cfg.resume_path
    if not resume.exists():
        raise FileNotFoundError(f"Resume not found: {resume}")
    raw = resume.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"{resume} is empty - copy the real resume into data/raw/ first.")

    docs = chunk(raw, cfg)
    if not docs:
        raise ValueError(
            "Chunking produced nothing. Check that the tags in resume.txt still match "
            "ingestion.structural.section_tags / item_tags in your config."
        )

    lengths = [len(d.page_content) for d in docs]
    report = BuildReport(
        index_path=cfg.display_path(cfg.index_path),
        index_fingerprint=cfg.index_fingerprint,
        strategy=cfg.ingestion.strategy,
        n_chunks=len(docs),
        by_type=summarize(docs),
        chars_total=sum(lengths),
        chars_mean=sum(lengths) // len(lengths),
        chars_max=max(lengths),
        elapsed_s=0.0,
    )

    over = [d for d in docs if len(d.page_content) > cfg.ingestion.split.max_chunk_chars]
    if over:
        report.warnings.append(
            f"{len(over)} chunk(s) exceed max_chunk_chars={cfg.ingestion.split.max_chunk_chars}; "
            f"largest is {max(len(d.page_content) for d in over)} chars."
        )
    if cfg.ingestion.strategy == "recursive":
        report.warnings.append(
            "The 'recursive' strategy produces no chunk_type metadata, so retrieval "
            "routing cannot fire and every question falls back to MMR. Compare against "
            "'structural' as pipeline-vs-pipeline, not chunker-vs-chunker."
        )

    if cfg.ingestion.dump_chunks and not dry_run:
        report.dumped_to = _dump_chunks(docs, cfg)

    if dry_run:
        report.dry_run = True
        report.elapsed_s = time.perf_counter() - started
        return report

    if cfg.index_path.exists() and not force:
        report.skipped = True
        report.elapsed_s = time.perf_counter() - started
        return report

    from langchain_community.vectorstores import FAISS

    emb = embedder if embedder is not None else build_embedder(cfg)
    store = FAISS.from_documents(docs, emb, distance_strategy=_faiss_distance(cfg))

    cfg.index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(cfg.index_path))
    _write_manifest(docs, cfg, report)

    report.elapsed_s = time.perf_counter() - started
    return report


def load_manifest(cfg: AppConfig) -> dict:
    """Read the manifest for this config's index. Raises if it was never built."""
    path = cfg.index_path / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No index manifest at {path}. Run: python scripts/ingest.py "
            f"--experiment {cfg.run.experiment_name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["build", "BuildReport", "load_manifest", "MANIFEST_NAME"]
