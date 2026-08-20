"""
ingest.py — Build the FAISS vector store for the portfolio chatbot.

CHUNKING STRATEGY (matches resume.txt)
------------------------------------------
resume.txt is structured with semantic XML tags + Markdown headers, so we
chunk on STRUCTURE instead of blind character windows:

  * Each innermost XML item becomes ONE chunk:
        <entry>, <project>, <institution>, <skill_category>,
        <certification_group>, <research_project>, <publication>,
        <presentation>, <role>, <recommendation>, <publication_item>
  * Leaf sections with no item children are emitted whole, one chunk each:
        <profile_summary>, <career_objective>, <about>
  * A section preamble (text before the first item, e.g. the "23 certifications"
    line) becomes a small "<section>_intro" chunk.
  * Tag attributes (company, role, period, name, provider, year, doi, url, ...)
    are carried into each chunk's metadata for richer / filtered retrieval.
  * A header-aware safeguard splits any unusually long chunk (> MAX_CHUNK_CHARS)
    while re-attaching its Markdown header block to every piece, so each piece
    stays self-contained and within the embedding model's effective window
    (all-MiniLM-L6-v2 ≈ 256 tokens).

Parsing is regex-based (stdlib `re`) rather than an XML parser, because the file
intentionally contains unescaped "&" and Markdown — a strict XML parser would
choke on those. The regex only matches well-formed tag markup, so characters
like "R² > 0.99" or ">40%" in the prose are left untouched.

Re-run this script whenever resume.txt changes:  python ingest.py
The output index (./faiss_db) is regenerated fresh each run.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- paths / config (anchored to this file so CWD doesn't matter) ---
BASE_DIR = Path(__file__).resolve().parent
RESUME_PATH = BASE_DIR / "resume.txt"
DB_PATH = BASE_DIR / "faiss_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Only split a structured chunk if it exceeds this many characters (~290 tokens).
# Most chunks are well under and pass through untouched.
MAX_CHUNK_CHARS = 1300

# Top-level section wrappers, processed in file order.
SECTION_TAGS = [
    "profile_summary", "career_objective", "about", "education",
    "experience", "projects", "research", "skills", "certifications",
    "newsletter", "conference_presentations", "leadership_volunteering",
    "recommendations",
]

# Innermost item tags that should each become their own chunk.
# NOTE: list longer prefixes first (publication_item before publication) so
# the regex alternation is greedy about the right tag.
ITEM_TAGS = [
    "entry", "project", "institution",
    "research_project", "publication_item", "publication",
    "skill_category", "certification_group",
    "presentation", "role", "recommendation",
]

# Friendly chunk_type label stored in metadata for each item tag.
TYPE_MAP = {
    "entry": "experience",
    "project": "project",
    "institution": "education",
    "research_project": "research_project",
    "publication_item": "newsletter",
    "publication": "publication",
    "skill_category": "skills",
    "certification_group": "certification_group",
    "presentation": "presentation",
    "role": "leadership_role",
    "recommendation": "recommendation",
}

# --- regex helpers ---
ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')
# Conservative: matches only well-formed tag-like markup, never bare "<"/">"
# in text such as "R² > 0.99" or ">40%".
TAG_RE = re.compile(r'</?[a-zA-Z_][\w-]*(?:\s+[\w:-]+\s*=\s*"[^"]*")*\s*/?>')
COMMENT_RE = re.compile(r'<!--.*?-->', re.DOTALL)
HEADER_RE = re.compile(r'^#{1,6}\s+(.*)$', re.MULTILINE)


def _block_re(tags):
    """Regex matching <tag ...attrs...> ... </tag> for any tag in `tags`."""
    names = "|".join(tags)
    return re.compile(
        rf'<({names})((?:\s+[\w:-]+\s*=\s*"[^"]*")*)\s*>(.*?)</\1>',
        re.DOTALL,
    )


SECTION_RE = _block_re(SECTION_TAGS)
ITEM_RE = _block_re(ITEM_TAGS)


def parse_attrs(attr_str):
    return {k: v for k, v in ATTR_RE.findall(attr_str or "")}


def clean_content(text):
    """Drop inline sub-tags (keep their text), collapse blank-line runs."""
    text = TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_header(text):
    m = HEADER_RE.search(text)
    return m.group(1).strip() if m else ""


def make_meta(section_name, chunk_type, content, attrs):
    """Tag attributes fill the metadata; our fixed keys always win."""
    meta = {k: v for k, v in attrs.items()}
    meta.update({
        "source": RESUME_PATH.name,
        "section": section_name,
        "chunk_type": chunk_type,
        "header": first_header(content),
    })
    return meta


def header_aware_split(doc, max_chars=MAX_CHUNK_CHARS):
    """Pass short docs through unchanged. For long ones, peel the leading
    Markdown header block and split only the body, re-attaching the header to
    every piece so each stays anchored and self-contained."""
    if len(doc.page_content) <= max_chars:
        return [doc]

    lines = doc.page_content.split("\n")
    h = 0
    while h < len(lines) and (lines[h].lstrip().startswith("#") or lines[h].strip() == ""):
        h += 1
    header_block = "\n".join(lines[:h]).strip()
    body = "\n".join(lines[h:]).strip()
    if not body:                      # nothing but headers — leave as-is
        return [doc]

    inner = max(300, max_chars - len(header_block) - 1)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=inner, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
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


def build_chunks(raw_text):
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_text = COMMENT_RE.sub("", raw_text)

    structured = []
    for sec in SECTION_RE.finditer(raw_text):
        section_name = sec.group(1)
        section_attrs = parse_attrs(sec.group(2))
        section_inner = sec.group(3)

        items = list(ITEM_RE.finditer(section_inner))

        # Leaf section -> one whole chunk.
        if not items:
            content = clean_content(section_inner)
            if content:
                meta = make_meta(section_name, section_name, content, section_attrs)
                structured.append(Document(page_content=content, metadata=meta))
            continue

        # Section preamble (text before the first item), if it has real content
        # beyond the "## Header" line -> a small intro chunk.
        preamble = clean_content(section_inner[: items[0].start()])
        if HEADER_RE.sub("", preamble).strip():
            meta = make_meta(section_name, f"{section_name}_intro", preamble, section_attrs)
            structured.append(Document(page_content=preamble, metadata=meta))

        # One chunk per item.
        for it in items:
            item_tag = it.group(1)
            item_attrs = parse_attrs(it.group(2))
            content = clean_content(it.group(3))
            if not content:
                continue
            chunk_type = TYPE_MAP.get(item_tag, item_tag)
            meta = make_meta(section_name, chunk_type, content, item_attrs)
            structured.append(Document(page_content=content, metadata=meta))

    # Header-aware safeguard for any oversized chunk.
    final = []
    for doc in structured:
        final.extend(header_aware_split(doc))
    return final


def main():
    raw = RESUME_PATH.read_text(encoding="utf-8")
    chunks = build_chunks(raw)

    if not chunks:
        raise SystemExit(
            "No chunks were produced — check that resume.txt's XML tags "
            "match the SECTION_TAGS / ITEM_TAGS lists in this script."
        )

    # NOTE: these embedding kwargs MUST match app.py's query embedder exactly
    # (normalize_embeddings=True). A mismatch makes stored vs. query vectors
    # incomparable and silently wrecks retrieval ranking.
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.from_documents(chunks, embedder)
    db.save_local(str(DB_PATH))

    print(f"✅ Stored {len(chunks)} chunks into {DB_PATH}")

    # --- summary by type ---
    by_type = {}
    for c in chunks:
        by_type[c.metadata["chunk_type"]] = by_type.get(c.metadata["chunk_type"], 0) + 1
    print("\nChunks by type:")
    for t, n in sorted(by_type.items()):
        print(f"  {t:24s} {n}")

    # --- debug: preview each chunk ---
    for i, c in enumerate(chunks):
        part = f" [part {c.metadata['part']}/{c.metadata['n_parts'] - 1}]" if "part" in c.metadata else ""
        print(f"\n--- Chunk {i} [{c.metadata['chunk_type']}]{part} ({len(c.page_content)} chars) ---")
        print(f"meta: {c.metadata}")
        print(c.page_content[:160].replace('\n', ' '))


if __name__ == "__main__":
    main()
