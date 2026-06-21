"""End-to-end verification (ROADMAP Phase 1 exit criterion).

Ingest the fixture corpus (md + transcript + html), then ask the canonical multi-hop
question. Assert the answer is cited and that the graph actually connects the two
notes via the Reranking → Retrieval relationship — something neither chunk states as
a single retrievable fact, but the *graph* does.
"""

from __future__ import annotations

from pathlib import Path

from secbrn.ingest.loaders import load_web

FIX = Path(__file__).parent / "fixtures"


def test_ingest_is_idempotent(brain):
    r1 = brain.ingest(FIX / "tools.md")
    assert r1.documents_ingested == 1
    r2 = brain.ingest(FIX / "tools.md")  # same content → skipped
    assert r2.documents_skipped == 1
    assert r2.documents_ingested == 0


def test_resolution_merges_pgvector_family(brain):
    brain.ingest(FIX / "tools.md")
    names = {e.name for e in brain.store.all_entities()}
    # PGVector / Pgvector variants should resolve to a single canonical pgvector node
    assert "pgvector" in names
    assert "PGVector" not in names and "Pgvector" not in names


def test_multihop_answer_with_citations_and_subgraph(brain):
    brain.ingest(FIX / "retrieval.md")
    brain.ingest(FIX / "rerankers.md")
    brain.ingest(FIX / "meeting.transcript.txt")
    brain.ingest_url("https://example.com/graphrag", html=(FIX / "article.html").read_text())

    ans = brain.ask("How does my note on retrieval relate to what I read about rerankers?")

    # grounded: has citations, not flagged uncited
    assert ans.citations, "answer must cite sources"
    assert not ans.uncited

    # the multi-hop fact lives in the graph: Reranking -IMPROVES-> Retrieval
    edges = {(e.subject, e.relation, e.object) for e in ans.bundle.edges}
    assert ("Reranking", "IMPROVES", "Retrieval") in edges

    # retrieval pulled from more than one source document
    titles = {c.document_title for c in ans.bundle.chunks}
    assert len(titles) >= 2


def test_stats_reports_graph(brain):
    brain.ingest(FIX / "retrieval.md")
    s = brain.stats()
    assert s["documents"] >= 1
    assert s["chunks"] >= 1
    assert s["entities"] >= 1
