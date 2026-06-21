"""Regression tests for ingest resilience (timeouts / flaky models)."""

from __future__ import annotations

import httpx
import pytest

from secbrn.config import Settings
from secbrn.graph.memory import InMemoryStore
from secbrn.pipeline import Brain
from secbrn.providers.fake import FakeEmbedder
from secbrn.providers.ollama import _OllamaClient
from tests.conftest import ScriptedLLM


def test_ollama_client_retries_transient_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):  # noqa: D401
            pass

        def json(self):
            return {"embedding": [0.0]}

    def fake_post(self, path, json):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("timed out")
        return _Resp()

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    c = _OllamaClient("http://x", timeout=1, max_retries=3, backoff=0)  # backoff 0 = fast
    out = c.post("/api/embeddings", {"model": "m", "prompt": "hi"})
    assert out == {"embedding": [0.0]}
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_ollama_client_gives_up_after_max_retries(monkeypatch):
    def always_timeout(self, path, json):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(httpx.Client, "post", always_timeout)
    c = _OllamaClient("http://x", timeout=1, max_retries=2, backoff=0)
    with pytest.raises(httpx.ReadTimeout):
        c.post("/api/embeddings", {"model": "m", "prompt": "hi"})


class _FlakyEmbedder(FakeEmbedder):
    """Times out on the 2nd chunk only — mimics a single slow PDF page."""

    def __init__(self, dim=64):
        super().__init__(dim=dim)
        self._n = 0

    def embed_one(self, text):
        self._n += 1
        if self._n == 2:
            raise httpx.ReadTimeout("timed out")
        return super().embed_one(text)


def test_one_chunk_timeout_does_not_abort_document(tmp_path):
    # a doc that produces several chunks
    doc = tmp_path / "big.md"
    body = "\n\n".join(f"## Section {i}\n" + ("word " * 120) for i in range(5))
    doc.write_text("# Big\n" + body, encoding="utf-8")

    s = Settings(provider="fake", embed_dim=64, chunk_size=300, chunk_overlap=40)
    llm = ScriptedLLM()
    brain = Brain(settings=s, store=InMemoryStore(), embedder=_FlakyEmbedder(64),
                  extract_llm=llm, answer_llm=llm)
    rep = brain.ingest(doc, resolve=False)

    assert rep.documents_ingested == 1
    assert rep.chunks_failed == 1          # exactly the flaky chunk was skipped
    assert rep.chunks_written >= 3         # the rest still ingested
    assert brain.stats()["chunks"] == rep.chunks_written
    brain.close()
