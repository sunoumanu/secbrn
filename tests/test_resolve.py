"""Stage 6 — entity resolution tests."""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.graph.memory import InMemoryStore
from secbrn.resolve.resolver import EntityResolver, string_similarity


def test_string_similarity():
    assert string_similarity("pgvector", "PGVector") == 1.0  # normalized identical
    assert string_similarity("Neo4j", "Neo4J") == 1.0
    assert string_similarity("cat", "dog") < 0.5


def _settings():
    return Settings(provider="fake", embed_dim=16)


def test_alias_seed_merges_pgvector_family():
    store = InMemoryStore()
    for name in ("pgvector", "Pgvector", "PGVector"):
        store.upsert_entity(name, "Tool", [])
    resolver = EntityResolver(_settings(), store)
    decisions = resolver.run()
    remaining = {e.name for e in store.all_entities()}
    assert remaining == {"pgvector"}
    canon = next(e for e in store.all_entities() if e.name == "pgvector")
    assert "PGVector" in canon.aliases and "Pgvector" in canon.aliases
    assert len(decisions) == 2


def test_fuzzy_merge_rewires_relations():
    store = InMemoryStore()
    store.upsert_entity("Reranking", "Concept", [])
    store.upsert_entity("Rerankng", "Concept", [])  # typo duplicate
    store.upsert_entity("Retrieval", "Concept", [])
    # correct spelling is more frequent -> it wins as canonical
    store.add_mention("c1", "Reranking")
    store.add_mention("c2", "Reranking")
    store.add_mention("c1", "Rerankng")
    store.add_relation("Rerankng", "IMPROVES", "Retrieval")
    resolver = EntityResolver(_settings(), store)
    resolver.run()
    names = {e.name for e in store.all_entities()}
    assert "Rerankng" not in names
    assert "Reranking" in names
    assert ("Reranking", "IMPROVES", "Retrieval") in store.relations
