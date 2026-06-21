"""Tests for rank-aware metrics + graph-aware retrieval scoring."""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.eval import metrics as M
from secbrn.graph.memory import InMemoryStore
from secbrn.models import Chunk, Document, Span
from secbrn.providers.fake import FakeEmbedder
from secbrn.retrieve.retriever import HybridRetriever


# ── new metrics ──────────────────────────────────────────────────────────────────
def test_r_precision_undistorted_by_k():
    # 1 relevant doc at rank 1, many irrelevant after: precision@R is perfect.
    retrieved = ["A", "B", "C", "D", "E", "F"]
    relevant = {"A"}
    assert M.r_precision(retrieved, relevant) == 1.0          # R=1, top-1 is A
    assert M.precision_at_k(retrieved, relevant, 6) < 0.2      # diluted by k


def test_average_precision_rewards_earlier_hits():
    rel = {"A", "B"}
    early = M.average_precision(["A", "B", "C"], rel)
    late = M.average_precision(["C", "A", "B"], rel)
    assert early == 1.0
    assert late < early


def test_ndcg_rewards_ranking_order():
    rel = {"A"}
    top = M.ndcg_at_k(["A", "B", "C"], rel, 3)
    low = M.ndcg_at_k(["B", "C", "A"], rel, 3)
    assert top == 1.0
    assert 0.0 < low < 1.0


# ── graph-aware scoring ────────────────────────────────────────────────────────────
def _store_with_graph():
    store = InMemoryStore()
    emb = FakeEmbedder(dim=64)
    # Two docs whose text is near-identical to the query, so vector/FT alone tie them.
    for did, title, text, ent in [
        ("d1", "Reranking Doc", "reranking helps", "Reranking"),
        ("d2", "Other Doc", "reranking helps", "Unrelated"),
    ]:
        store.upsert_document(Document(id=did, source_type="markdown", uri=f"file://{did}", title=title, raw_text=text))
        c = Chunk(id=f"{did}:0", document_id=did, position=0, text=text, span=Span("line", 1, 1),
                  embedding=emb.embed_one(text), embed_model="fake", embed_dim=64)
        store.upsert_chunk(c)
        store.upsert_entity(ent, "Concept", [])
        store.add_mention(c.id, ent)
    # graph: query mentions Reranking, connected to Retrieval
    store.upsert_entity("Retrieval", "Concept", [])
    store.add_relation("Reranking", "IMPROVES", "Retrieval")
    return store, emb


def test_graph_boost_promotes_connected_chunk():
    store, emb = _store_with_graph()
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2, retrieve_hops=2, graph_boost=1.0)
    r = HybridRetriever(s, store, emb)
    hits = r.search("reranking and retrieval")
    # d1 (entity Reranking, in the query neighbourhood) should rank first
    assert hits[0].document_title == "Reranking Doc"
    assert "graph" in hits[0].via


def test_graph_boost_off_is_noop():
    store, emb = _store_with_graph()
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2, retrieve_hops=2, graph_boost=0.0)
    r = HybridRetriever(s, store, emb)
    hits = r.search("reranking and retrieval")
    assert all("graph" not in h.via for h in hits)


def test_fuzzy_match_entities_seeds_substring():
    store = InMemoryStore()
    store.upsert_entity("GraphRAG", "Topic", [])
    store.upsert_entity("Ollama", "Tool", [])
    seeds = store.match_entities("multi-hop reasoning over a knowledge graph")
    assert "GraphRAG" in seeds          # 'graph' fuzzily seeds 'GraphRAG'
    assert "Ollama" not in seeds


def _two_docs(title_a, ent_a, title_b, ent_b, text="alpha beta gamma"):
    store = InMemoryStore()
    emb = FakeEmbedder(dim=64)
    for did, title, ent in [("d1", title_a, ent_a), ("d2", title_b, ent_b)]:
        store.upsert_document(Document(id=did, source_type="markdown", uri=f"file://{did}", title=title, raw_text=text))
        c = Chunk(id=f"{did}:0", document_id=did, position=0, text=text, span=Span("line", 1, 1),
                  embedding=emb.embed_one(text), embed_model="fake", embed_dim=64)
        store.upsert_chunk(c)
        store.upsert_entity(ent, "Concept", [])
        store.add_mention(c.id, ent)
    return store, emb


def test_title_boost_promotes_title_match():
    # identical text + no graph signal; only the title differs
    store, emb = _two_docs("GraphRAG", "X", "Misc Notes", "Y")
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2, graph_boost=0.0, title_boost=1.0)
    r = HybridRetriever(s, store, emb)
    hits = r.search("knowledge graph reasoning")  # 'graph' ~ 'graphrag' title
    assert hits[0].document_title == "GraphRAG"
    assert "title" in hits[0].via


def test_seed_hit_beats_neighbor_only():
    # docA mentions the seed entity; docB mentions only a 1-hop neighbour
    store, emb = _two_docs("A", "Reranking", "B", "Retrieval")
    store.add_relation("Reranking", "IMPROVES", "Retrieval")  # Retrieval is a neighbour
    s = Settings(provider="fake", embed_dim=64, retrieve_top_k=2, retrieve_hops=2,
                 graph_boost=1.0, title_boost=0.0)
    r = HybridRetriever(s, store, emb)
    hits = r.search("reranking")  # seed = Reranking
    assert hits[0].document_title == "A"  # seed hit (weight 1) beats neighbour (weight 0.25)


def test_fake_embedder_auto_dim_sentinel():
    # dim=0 (auto sentinel from A/B runs) must still yield a usable vector
    e = FakeEmbedder(dim=0)
    v = e.embed_one("hello world")
    assert e.dim == 256 and len(v) == 256


def test_inmemory_recreate_vector_index_is_noop():
    from secbrn.graph.memory import InMemoryStore
    s = InMemoryStore()
    s.ensure_schema()
    s.recreate_vector_index()  # must not raise
    assert s.indexes_present()["chunk_vec"] is True


def test_embed_model_ab_runs_offline():
    """A/B over embedding models works in the dim-agnostic in-memory store."""
    from pathlib import Path
    from secbrn.config import get_settings
    from secbrn.eval import Evaluator, load_goldset
    from secbrn.pipeline import Brain

    gold = Path(__file__).parent.parent / "eval" / "gold.json"
    gs = load_goldset(gold)
    base = get_settings().model_copy(update={"provider": "fake"})
    reports = []
    for m in ("nomic-embed-text", "all-minilm"):
        s2 = base.model_copy(update={"embed_model": m, "embed_dim": 0})
        brain = Brain.isolated(s2)
        try:
            brain.ingest(gs.corpus_path(), resolve=False)
            reports.append(Evaluator(brain).evaluate(gs))
        finally:
            brain.close()
    assert all(r.retrieval is not None for r in reports)
