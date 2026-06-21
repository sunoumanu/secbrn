"""Tests for store.clear() and the `secbrn reset` plumbing."""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.pipeline import Brain
from secbrn.graph.memory import InMemoryStore


def _brain():
    from secbrn.providers.fake import FakeEmbedder, FakeLLM
    s = Settings(provider="fake", graph_backend="memory")
    return Brain(settings=s, store=InMemoryStore(), embedder=FakeEmbedder(dim=64),
                 extract_llm=FakeLLM(), answer_llm=FakeLLM())


def test_clear_empties_store(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Vectors\n\nNeo4j relates to Ollama and pgvector for retrieval.", encoding="utf-8")
    b = _brain()
    b.ingest(p)
    before = b.stats()
    assert before["documents"] >= 1 and before["chunks"] >= 1

    b.store.clear()
    after = b.stats()
    assert after["documents"] == 0
    assert after["chunks"] == 0
    assert after["entities"] == 0
    assert after["relations"] == 0
    assert after["mentions"] == 0


def test_clear_then_reingest_not_skipped(tmp_path):
    """After clear(), the same file ingests again instead of dedup-skipping."""
    p = tmp_path / "note.md"
    p.write_text("# Vectors\n\nNeo4j relates to Ollama.", encoding="utf-8")
    b = _brain()
    r1 = b.ingest(p)
    assert r1.documents_ingested == 1

    r2 = b.ingest(p)  # unchanged -> skipped
    assert r2.documents_skipped == 1 and r2.documents_ingested == 0

    b.store.clear()
    r3 = b.ingest(p)  # clean slate -> ingested again
    assert r3.documents_ingested == 1 and r3.documents_skipped == 0


def test_store_clear_is_idempotent():
    b = _brain()
    b.store.clear()
    b.store.clear()
    assert b.stats()["documents"] == 0
