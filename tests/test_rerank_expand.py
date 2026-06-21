"""Tests for query expansion + LLM reranking (Stage 7 / 7.5)."""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.graph.memory import InMemoryStore
from secbrn.models import Chunk, Document, RetrievedChunk, Span
from secbrn.providers.fake import FakeEmbedder
from secbrn.retrieve.expand import expand_query
from secbrn.retrieve.rerank import LLMReranker
from secbrn.retrieve.retriever import HybridRetriever


class _StubLLM:
    def __init__(self, out): self._out = out; self.model = "stub"
    def complete(self, prompt, *, system=None, temperature=0.0): return self._out
    def complete_json(self, prompt, *, system=None): return self._out


# ── query expansion ──────────────────────────────────────────────────────────────
def test_expand_query_appends_terms():
    llm = _StubLLM("nearest neighbour, ANN, similarity search")
    out = expand_query("vector search", llm, n_terms=6)
    assert out.startswith("vector search ")
    assert "nearest neighbour" in out and "ANN" in out


def test_expand_query_fallback_on_error():
    class Boom:
        model = "boom"
        def complete(self, *a, **k): raise RuntimeError("down")
        def complete_json(self, *a, **k): raise RuntimeError("down")
    assert expand_query("hello world", Boom()) == "hello world"


# ── reranker ──────────────────────────────────────────────────────────────────────
def _chunks(n):
    out = []
    for i in range(n):
        out.append(RetrievedChunk(
            chunk_id=f"c{i}", document_id="d", document_title=f"T{i}", uri="",
            text=f"passage {i}", span=Span("line", 1, 1), score=float(n - i), via="vector"))
    return out


def test_reranker_reorders_by_model_output():
    chunks = _chunks(3)  # initial order c0,c1,c2
    rr = LLMReranker(_StubLLM("[3,1,2]"))   # -> head[2], head[0], head[1] == c2,c0,c1
    out = rr.rerank("q", chunks, top_n=3)
    assert [c.chunk_id for c in out] == ["c2", "c0", "c1"]
    assert "rerank" in out[0].via


def test_reranker_fallback_keeps_order_on_garbage():
    chunks = _chunks(3)
    rr = LLMReranker(_StubLLM("the passages are all great"))
    out = rr.rerank("q", chunks, top_n=3)
    assert [c.chunk_id for c in out] == ["c0", "c1", "c2"]


# ── integration: rerank changes retrieval order ──────────────────────────────────
def _store_two():
    store = InMemoryStore()
    emb = FakeEmbedder(dim=64)
    for did, title, text in [("d1", "A", "alpha alpha"), ("d2", "B", "alpha alpha")]:
        store.upsert_document(Document(id=did, source_type="markdown", uri=f"file://{did}", title=title, raw_text=text))
        c = Chunk(id=f"{did}:0", document_id=did, position=0, text=text, span=Span("line", 1, 1),
                  embedding=emb.embed_one(text), embed_model="fake", embed_dim=64)
        store.upsert_chunk(c)
    return store, emb


def test_retriever_applies_rerank_when_enabled():
    store, emb = _store_two()
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2, graph_boost=0.0,
                 title_boost=0.0, rerank=True, rerank_candidates=2)
    # reranker that forces the 2nd candidate to the top
    rr = LLMReranker(_StubLLM("[2,1]"))
    r = HybridRetriever(s, store, emb, llm=_StubLLM(""), reranker=rr)
    hits = r.search("alpha")
    assert "rerank" in hits[0].via
    assert len(hits) == 2
