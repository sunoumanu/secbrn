"""Ingest progress callback emits sane, monotonic events."""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.pipeline import Brain, IngestProgress
from secbrn.graph.memory import InMemoryStore
from secbrn.providers.fake import FakeEmbedder, FakeLLM


def _brain(level=4):
    s = Settings(provider="fake", graph_backend="memory", ingest_concurrency=level)
    return Brain(settings=s, store=InMemoryStore(), embedder=FakeEmbedder(dim=64),
                 extract_llm=FakeLLM(), answer_llm=FakeLLM())


def test_progress_events_cover_both_phases(tmp_path):
    p = tmp_path / "big.md"
    p.write_text("# Title\n\n" + ("Neo4j relates to Ollama and pgvector. " * 200), encoding="utf-8")

    events: list[IngestProgress] = []
    b = _brain()
    rep = b.ingest(p, progress=events.append)

    phases = {e.phase for e in events}
    assert phases == {"embed", "extract"}

    embed = [e for e in events if e.phase == "embed"]
    # starts at 0, ends at total, monotonic non-decreasing
    assert embed[0].done == 0
    assert embed[-1].done == embed[-1].total == rep.chunks_written
    assert all(b1.done <= b2.done for b1, b2 in zip(embed, embed[1:]))
    assert rep.chunks_written > 2  # a real multi-chunk doc


def test_progress_none_is_fine(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("# T\n\nNeo4j and Ollama.", encoding="utf-8")
    b = _brain(level=1)
    rep = b.ingest(p)  # no progress callback
    assert rep.chunks_written >= 1
