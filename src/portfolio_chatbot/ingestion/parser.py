"""
parser.py - resume.txt -> structured items.

Turns the semantically tagged resume into a flat, ordered list of `ParsedItem`
records. It does NOT decide chunk sizes; that is `chunker.py`'s job. Keeping
the two apart is what lets a chunking experiment change how text is split
without touching how the document is understood.

Parsing is regex-based rather than XML-based, for the reason the original
`ingest.py` documented: the file deliberately contains unescaped `&` and
Markdown, which a strict XML parser rejects. The expressions here only match
well-formed tag markup, so prose like "R2 > 0.99" or ">40%" is left alone.

Every tag name, the type map, and the cleanup switches come from
`config.ingestion.structural` - nothing is hardcoded.

Three kinds of item come out, and the distinction matters for retrieval
because `chunk_type` is what routing filters on:

* ``item``          - one innermost tagged record (`<entry>`, `<project>`, ...).
                      chunk_type comes from the config's type_map.
* ``leaf_section``  - a section with no item children (`<profile_summary>`,
                      `<about>`), emitted whole. chunk_type is the section name.
* ``section_intro`` - the prose before a section's first item, when there is
                      any beyond the heading. chunk_type is `<section>_intro`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..config import AppConfig

ItemKind = Literal["item", "leaf_section", "section_intro"]

# Matches key="value" pairs inside a tag.
ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')

# Conservative on purpose: only well-formed tag markup, never a bare "<" or
# ">" appearing in prose.
TAG_RE = re.compile(r'</?[a-zA-Z_][\w-]*(?:\s+[\w:-]+\s*=\s*"[^"]*")*\s*/?>')

COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HEADER_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class ParsedItem:
    """One semantic unit of the resume, before any size-based splitting."""
    section: str
    chunk_type: str
    kind: ItemKind
    content: str
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def header(self) -> str:
        """First Markdown heading inside the content, if any."""
        m = HEADER_RE.search(self.content)
        return m.group(1).strip() if m else ""


def _block_re(tags: list[str]) -> re.Pattern[str]:
    """Regex matching ``<tag ...attrs...> ... </tag>`` for any tag in `tags`.

    Names are sorted longest-first so an alternation like `publication` can
    never shadow `publication_item`. (Backtracking would recover anyway, since
    `publication` must be followed by whitespace or `>`, but relying on that is
    a needless footgun for whoever edits the tag list next.)
    """
    names = "|".join(sorted(tags, key=len, reverse=True))
    return re.compile(
        rf'<({names})((?:\s+[\w:-]+\s*=\s*"[^"]*")*)\s*>(.*?)</\1>',
        re.DOTALL,
    )


def parse_attrs(attr_str: str | None) -> dict[str, str]:
    return {k: v for k, v in ATTR_RE.findall(attr_str or "")}


def clean_content(text: str, *, collapse_blank_lines: bool = True) -> str:
    """Drop inline sub-tags but keep their text, then tidy whitespace.

    `<contact>` and `<links>` inside `<profile_summary>` are not item tags, so
    they are stripped here and their content survives as part of the section's
    single chunk. That is intentional - contact details stay searchable.
    """
    text = TAG_RE.sub("", text)
    if collapse_blank_lines:
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_markup(raw: str, cfg: AppConfig) -> str:
    """All tags removed, whitespace tidied. Used by the `recursive` strategy."""
    s = cfg.ingestion.structural
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if s.strip_html_comments:
        text = COMMENT_RE.sub("", text)
    return clean_content(text, collapse_blank_lines=s.collapse_blank_lines)


def parse(raw: str, cfg: AppConfig) -> list[ParsedItem]:
    """Parse the resume into ordered `ParsedItem`s.

    Sections are walked in file order, and items within each section in their
    own order, so the output is deterministic. `chunker.py` relies on that for
    stable chunk ids.
    """
    s = cfg.ingestion.structural

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    if s.strip_html_comments:
        text = COMMENT_RE.sub("", text)

    section_re = _block_re(list(s.section_tags))
    item_re = _block_re(list(s.item_tags))
    collapse = s.collapse_blank_lines

    out: list[ParsedItem] = []

    for sec in section_re.finditer(text):
        section_name = sec.group(1)
        section_attrs = parse_attrs(sec.group(2)) if s.carry_tag_attributes_to_metadata else {}
        inner = sec.group(3)

        items = list(item_re.finditer(inner))

        # No item children -> the whole section is one chunk.
        if not items:
            content = clean_content(inner, collapse_blank_lines=collapse)
            if content:
                out.append(ParsedItem(
                    section=section_name,
                    chunk_type=section_name,
                    kind="leaf_section",
                    content=content,
                    attrs=section_attrs,
                ))
            continue

        # Prose before the first item, if it is more than just the heading.
        preamble = clean_content(inner[: items[0].start()], collapse_blank_lines=collapse)
        if HEADER_RE.sub("", preamble).strip():
            out.append(ParsedItem(
                section=section_name,
                chunk_type=f"{section_name}_intro",
                kind="section_intro",
                content=preamble,
                attrs=section_attrs,
            ))

        for it in items:
            tag = it.group(1)
            attrs = parse_attrs(it.group(2)) if s.carry_tag_attributes_to_metadata else {}
            content = clean_content(it.group(3), collapse_blank_lines=collapse)
            if not content:
                continue
            out.append(ParsedItem(
                section=section_name,
                chunk_type=s.type_map.get(tag, tag),
                kind="item",
                content=content,
                attrs=attrs,
            ))

    return out


__all__ = ["ParsedItem", "parse", "strip_markup", "clean_content", "parse_attrs", "HEADER_RE"]
