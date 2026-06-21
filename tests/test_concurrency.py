"""Tests for the I/O concurrency added to ingest / resolve / retrieve.

We assert three things that matter for correctness *and* speed:
  1. ``map_workers`` preserves input order and turns exceptions into data.
  2. Ingest actually runs embed/extract calls concurrently (wall-clock overlap),
     not just nominally.
  3. Concurrency changes latency, never results: a parallel ingest yields the same
     graph as a strictly-sequential one.
"""

from __future__ import annotations

import time

import pytest

from secbrn.config import Settings
from secbrn.pipeline import Brain
from secbrn.util import map_workers


# ── map_workers ──────────────────────────────────────────────────────────────────
def test_map_workers_preserves_order_and_captures_errors():
    def f(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x * 10

    out = map_workers(f, [0, 1, 2, 3], workers=4)
    assert [r for r, _ in out] == [0, 10, None, 30]
    assert out[2][0] is None and isinstance(out[2][1], ValueError)
    # every other slot has no exception
    assert all(e is None for i, (_, e) in enumerate(out) if i != 2)


def test_map_workers_inline_when_serial():
    assert map_workers(lambda x: x + 1, [1, 2, 3], workers=1) == [(2, None), (3, None), (4, None)]
    assert map_workers(lambda x: x, [], workers=8) == []


def test_map_workers_actually_overlaps():
    """N sleeping calls at concurrency N finish in ~1 unit, not N units."""
    def slow(_):
        time.sleep(0.1)
        return 1

    t0 = time.perf_counter()
    map_workers(slow, list(range(8)), workers=8)
    parallel = time.perf_counter() - t0

    t0 = time.perf_counter()
    map_workers(slow, list(range(8)), workers=1)
    serial = time.perf_counter() - t0

    # 8x work; parallel should be dramatically faster than serial.
    assert parallel < serial / 3


# ── ingest: overlap + determinism ────────────────────────────────────────────────
class _SlowEmbedder:
    """Fake embedder that sleeps per call, to measure overlap."""

    model = "slow-embed"
    dim = 8

    def __init__(self, delay=0.05):
        self.delay = delay

    def embed_one(self, text: str):
        time.sleep(self.delay)
        # deterministic tiny vector
        v = [0.0] * self.dim
        v[len(text) % self.dim] = 1.0
        return v

    def embed(self, texts):
        return [self.embed_one(t) for t in texts]


def _doc_with_many_chunks(tmp_path, n_chars=6000):
    p = tmp_path / "big.md"
    # enough text to produce several chunks at the default chunk size
    p.write_text("# Title\n\n" + ("Vector graphs relate to Neo4j and Ollama. " * (n_chars // 40)),
                 encoding="utf-8")
    return p


def _brain(settings: Settings) -> Brain:
    from secbrn.graph.memory import InMemoryStore
    from secbrn.providers.fake import FakeLLM

    return Brain(
        settings=settings,
        store=InMemoryStore(),
        embedder=_SlowEmbedder(),
        extract_llm=FakeLLM(),
        answer_llm=FakeLLM(),
    )


def test_ingest_embeds_concurrently(tmp_path):
    doc = _doc_with_many_chunks(tmp_path)

    s_par = Settings(provider="fake", graph_backend="memory", ingest_concurrency=8)
    b = _brain(s_par)
    t0 = time.perf_counter()
    rep_par = b.ingest(doc)
    par = time.perf_counter() - t0

    s_seq = Settings(provider="fake", graph_backend="memory", ingest_concurrency=1)
    b2 = _brain(s_seq)
    t0 = time.perf_counter()
    rep_seq = b2.ingest(doc)
    seq = time.perf_counter() - t0

    assert rep_par.chunks_written > 2, "need several chunks to see overlap"
    assert rep_par.chunks_written == rep_seq.chunks_written
    # parallel embedding of N chunks should be meaningfully faster
    assert par < seq * 0.7


def test_concurrency_is_result_invariant(tmp_path):
    """Same graph regardless of concurrency level."""
    doc = _doc_with_many_chunks(tmp_path)

    def run(level):
        s = Settings(provider="fake", graph_backend="memory",
                     ingest_concurrency=level, resolve_concurrency=level)
        b = _brain(s)
        rep = b.ingest(doc)
        return (rep.chunks_written, rep.entities_extracted,
                rep.relations_extracted, rep.merges, b.stats())

    assert run(1) == run(8)
