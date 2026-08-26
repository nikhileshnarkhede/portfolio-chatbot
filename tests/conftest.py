"""
Shared fixtures.

The point here is that nodes are testable without a network, an API key, or a
model download. `fake_index` builds a real FAISS store from the real resume
using a deterministic fake embedder, and `scripted_llm` replaces the chat model
with one that returns whatever the test wants - including raising rate-limit
errors on demand, which is how the fallback chain gets covered without waiting
for Groq's free tier to run out.
"""

from __future__ import annotations

import shutil

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from portfolio_chatbot.config import load_config
from portfolio_chatbot.nodes import CONFIG_KEY

EMBED_DIM = 384


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def runnable_config(cfg):
    """The RunnableConfig shape every node expects."""
    return {"configurable": {CONFIG_KEY: cfg, "thread_id": "test"}}


@pytest.fixture
def fake_embedder():
    return DeterministicFakeEmbedding(size=EMBED_DIM)


@pytest.fixture
def fake_index(cfg, fake_embedder, monkeypatch):
    """A real FAISS store over the real resume, built with fake vectors.

    Registered directly in the retriever's cache, so no index needs to exist on
    disk and no embedding model is downloaded.
    """
    from langchain_community.vectorstores import FAISS

    from portfolio_chatbot.ingestion.chunker import chunk
    from portfolio_chatbot.tools import retriever_tool

    resume = cfg.resume_path
    if not resume.exists() or not resume.read_text(encoding="utf-8").strip():
        pytest.skip("data/raw/resume.txt is missing or empty")

    docs = chunk(resume.read_text(encoding="utf-8"), cfg)
    store = FAISS.from_documents(docs, fake_embedder)

    # The store is cached in memory, but `ui/app.py` refuses to run when
    # `cfg.index_path` does not EXIST - it shows "no vector index" and returns
    # before drawing anything. So the directory has to be there too.
    #
    # Creating it here is what makes these tests independent of the machine.
    # They used to pass only because an index directory left over from a real
    # `scripts/ingest.py` happened to match the current fingerprint - so a
    # change to what the fingerprint covers turned six green tests red without
    # a line of app code changing.
    index_dir = cfg.index_path
    created = not index_dir.exists()
    index_dir.mkdir(parents=True, exist_ok=True)

    retriever_tool.clear_cache()
    retriever_tool.register_store(cfg, store)
    yield store
    retriever_tool.clear_cache()

    if created:
        shutil.rmtree(index_dir, ignore_errors=True)


@pytest.fixture
def existing_index(cfg):
    """An empty index directory for the CURRENT fingerprint.

    Several checks ask only "does this index exist" - `ui/app.py` before it
    draws anything, `eval.apply.check_ready` before it promotes a config. Both
    are satisfied by the directory being there, and neither should be satisfied
    by whatever an earlier real ingest happened to leave on this machine.
    """
    index_dir = cfg.index_path
    created = not index_dir.exists()
    index_dir.mkdir(parents=True, exist_ok=True)
    yield index_dir
    if created:
        shutil.rmtree(index_dir, ignore_errors=True)


class ScriptedLLM:
    """Stands in for ChatGroq. Returns canned text, or raises on demand."""

    def __init__(self, model: str, behaviour: dict):
        self.model = model
        self.behaviour = behaviour

    def _resolve(self):
        outcome = self.behaviour.get(self.model, self.behaviour.get("*", "ok"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def invoke(self, _prompt):
        return AIMessage(content=self._resolve())

    def stream(self, _prompt):
        for word in self._resolve().split(" "):
            yield AIMessage(content=word + " ")


@pytest.fixture
def scripted_llm(monkeypatch):
    """Install a scripted model. Call the returned function with a behaviour map.

        scripted_llm({"openai/gpt-oss-20b": RateLimitError(), "*": "hello"})

    Keys are model names; "*" is the default. An Exception value is raised
    instead of returned, which is how rate-limit fallback gets exercised.
    """
    def install(behaviour: dict):
        def build(cfg, model=None):
            return ScriptedLLM(model or cfg.llm.primary_model, behaviour)

        for module in ("generate", "summarize", "expand_query"):
            monkeypatch.setattr(
                f"portfolio_chatbot.nodes.{module}.build_llm", build, raising=False
            )
        monkeypatch.setattr("portfolio_chatbot.llm.provider.build_llm", build)
        return build

    return install


@pytest.fixture
def rate_limit_error():
    return lambda: RuntimeError("Error code: 429 - rate_limit_exceeded")
