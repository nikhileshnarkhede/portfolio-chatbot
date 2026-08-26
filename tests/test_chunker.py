"""
Tests for parsing and chunking.

The centrepiece is `test_matches_verified_baseline_snapshot`. During the port,
the new `structural` strategy was compared chunk-for-chunk against the original
`ingest.py` on the real resume: 61 chunks, zero content differences, zero
metadata differences. `tests/fixtures/baseline_chunks.json` freezes that
verified output. The old script is not a dependency of the test suite, so the
snapshot is what keeps the baseline honest - if a refactor changes chunking,
this fails, and the numbers in `runs/` stop being comparable to earlier ones.

If a change to chunking is intentional, regenerate the fixture deliberately and
treat every stored eval result as belonging to the previous chunking regime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_chatbot.config import load_config
from portfolio_chatbot.ingestion.chunker import (
    IMPLEMENTED,
    chunk,
    chunk_recursive,
    chunk_structural,
    summarize,
)
from portfolio_chatbot.ingestion.parser import parse, strip_markup

FIXTURE = Path(__file__).parent / "fixtures" / "baseline_chunks.json"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw(cfg):
    path = cfg.resume_path
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        pytest.skip("data/raw/resume.txt is missing or empty")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def docs(raw, cfg):
    return chunk_structural(raw, cfg)


# ---------------------------------------------------------------- parser

def test_parse_finds_every_section(raw, cfg):
    sections = {i.section for i in parse(raw, cfg)}
    for expected in ("experience", "projects", "skills", "education", "research"):
        assert expected in sections


def test_leaf_sections_are_emitted_whole(raw, cfg):
    kinds = {i.section: i.kind for i in parse(raw, cfg) if i.kind == "leaf_section"}
    assert "about" in kinds


def test_contact_details_survive_tag_stripping(raw, cfg):
    """`<contact>` is not an item tag; its text must still reach the index."""
    profile = [i for i in parse(raw, cfg) if i.section == "profile_summary"]
    assert profile, "profile_summary section not parsed"
    assert "<contact>" not in profile[0].content
    assert "@" in profile[0].content  # the email survived


def test_prose_comparison_operators_are_not_treated_as_tags(cfg):
    """'R2 > 0.99' and '>40%' must not be eaten by the tag regex."""
    text = "<about>\nAccuracy R2 > 0.99 and >40% faster.\n</about>"
    items = parse(text, cfg)
    assert len(items) == 1
    assert "R2 > 0.99" in items[0].content
    assert ">40%" in items[0].content


def test_strip_markup_removes_all_tags(raw, cfg):
    assert "<project" not in strip_markup(raw, cfg)


# ---------------------------------------------------------------- baseline parity

def test_matches_verified_baseline_snapshot(docs):
    """REGRESSION: structural chunking must stay identical to the verified port."""
    expected = json.loads(FIXTURE.read_text())
    assert len(docs) == expected["n_chunks"]

    import hashlib
    actual = [
        {
            "chunk_id": d.metadata["chunk_id"],
            "chunk_type": d.metadata["chunk_type"],
            "section": d.metadata["section"],
            "part": d.metadata["part"],
            "chars": len(d.page_content),
            "sha1": hashlib.sha1(d.page_content.encode()).hexdigest()[:12],
        }
        for d in docs
    ]
    assert actual == expected["chunks"]


# ---------------------------------------------------------------- chunk ids

def test_chunk_ids_are_unique(docs):
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(set(ids)) == len(ids)


def test_chunk_ids_are_deterministic(raw, cfg):
    a = [d.metadata["chunk_id"] for d in chunk_structural(raw, cfg)]
    b = [d.metadata["chunk_id"] for d in chunk_structural(raw, cfg)]
    assert a == b


def test_editing_content_preserves_chunk_ids(raw, cfg):
    """A typo fix must not invalidate golden-set references."""
    before = [d.metadata["chunk_id"] for d in chunk_structural(raw, cfg)]
    edited = raw.replace("Machine Learning has", "Machine learning has")
    after = [d.metadata["chunk_id"] for d in chunk_structural(edited, cfg)]
    assert before == after


def test_renaming_an_item_changes_only_its_id(raw, cfg):
    before = {d.metadata["chunk_id"] for d in chunk_structural(raw, cfg)}
    renamed = raw.replace('name="Supply Chain Tracker"', 'name="Supply Chain Monitor"')
    after = {d.metadata["chunk_id"] for d in chunk_structural(renamed, cfg)}
    assert len(before ^ after) == 2  # exactly one id out, one in


# ---------------------------------------------------------------- splitting

def test_no_chunk_exceeds_the_configured_limit(docs, cfg):
    limit = cfg.ingestion.split.max_chunk_chars
    assert all(len(d.page_content) <= limit for d in docs)


def test_split_pieces_reattach_their_header(docs):
    """Piece 2 of an item must still say which item it belongs to."""
    for d in docs:
        if d.metadata["n_parts"] > 1 and d.metadata["header"]:
            assert d.metadata["header"] in d.page_content


def test_smaller_max_chunk_chars_yields_more_chunks(raw):
    big = chunk_structural(raw, load_config())
    small = chunk_structural(raw, load_config("exp002_chunk_512"))
    assert len(small) > len(big)


def test_split_pieces_are_numbered_consistently(docs):
    for d in docs:
        assert 0 <= d.metadata["part"] < d.metadata["n_parts"]


# ---------------------------------------------------------------- strategies

def test_recursive_produces_no_routable_types(raw, cfg):
    """Documented consequence: naive chunking loses the routing metadata."""
    assert set(summarize(chunk_recursive(raw, cfg))) == {"text"}


def test_structural_preserves_routable_types(docs):
    types = set(summarize(docs))
    assert {"project", "experience", "skills", "education"} <= types


def test_strategy_is_selected_by_config(raw):
    rec = load_config(overrides=['ingestion.strategy="recursive"'])
    assert set(summarize(chunk(raw, rec))) == {"text"}


@pytest.mark.parametrize("name", ["fixed", "semantic"])
def test_unimplemented_strategies_fail_loudly(raw, name):
    """A silent fallback would produce numbers you cannot attribute."""
    cfg = load_config(overrides=[f'ingestion.strategy="{name}"'])
    with pytest.raises(NotImplementedError, match=name):
        chunk(raw, cfg)


def test_implemented_set_matches_reality(raw, cfg):
    for name in IMPLEMENTED:
        assert chunk(raw, load_config(overrides=[f'ingestion.strategy="{name}"']))


# ---------------------------------------------------------------- metadata

def test_every_chunk_carries_the_retrieval_metadata(docs):
    for d in docs:
        for key in ("chunk_id", "chunk_type", "section", "header", "identity", "part", "n_parts", "source"):
            assert key in d.metadata, f"missing {key}"


def test_tag_attributes_reach_metadata(docs):
    projects = [d for d in docs if d.metadata["chunk_type"] == "project"]
    assert projects
    assert any(d.metadata.get("url") for d in projects)
