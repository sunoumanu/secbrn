"""Stage 7 — hybrid retrieval tests (in-memory store)."""

from __future__ import annotations

from secbrn.graph.memory import InMemoryStore
from secbrn.models import Chunk, Document, Span
from secbrn.providers.fake import FakeEmbedder
from secbrn.retrieve.retriever import HybridRetriever
from secbrn.config import Settings


def _seed():
    store = InMemoryStore()
    emb = FakeEmbedder(dim=64)
    docs = {
        "d1": ("Retrieval Notes", "Reranking improves retrieval quality in RAG."),
        "d2": ("Tooling", "Neo4j is a graph database with a vector index."),
    }
    for did, (title, text) in docs.items():
        store.upsert_document(Document(id=did, source_type="markdown", uri=f"file://{did}", title=title, raw_text=text))
        c = Chunk(id=f"{did}:0", document_id=did, position=0, text=text, span=Span("line", 1, 1),
                  embedding=emb.embed_one(text), embed_model="fake", embed_dim=64)
        store.upsert_chunk(c)
        store.upsert_entity(title, "Concept", [])
        store.add_mention(c.id, title)
    return store, emb


def test_vector_and_fulltext_fusion_ranks_relevant_first():
    store, emb = _seed()
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2)
    r = HybridRetriever(s, store, emb)
    hits = r.search("how does reranking help retrieval")
    assert hits
    assert hits[0].document_title == "Retrieval Notes"
    assert "+" in hits[0].via or hits[0].via in ("vector", "fulltext")


def test_retrieve_bundle_has_chunks_and_entities():
    store, emb = _seed()
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2, retrieve_hops=1)
    r = HybridRetriever(s, store, emb)
    bundle = r.retrieve("reranking retrieval")
    assert bundle.chunks
    assert "Retrieval Notes" in bundle.entities
