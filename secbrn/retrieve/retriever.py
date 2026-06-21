"""Stage 7 — Hybrid retrieval: vector + full-text + graph + lexical signals.

Pipeline per query:
  1. Vector search over chunk embeddings.
  2. Full-text / keyword search over chunk text.
  3. Reciprocal-rank fusion of (1) and (2).
  4. Re-score the fused candidates with two cheap, high-signal boosts:
       - graph boost: chunks that mention the query's *seed* entities (the entities the
         query is literally about) are boosted strongly; chunks that only mention distant
         graph neighbours get a small boost. This rewards relevance, not entity density.
       - title boost: chunks whose document title / section heading shares terms with the
         query are boosted (a doc titled "GraphRAG" should win a "knowledge graph" query).
  5. Graph expansion from the top chunks' entities builds the answer's subgraph.

Interface matches neo4j-graphrag retrievers so they remain drop-in (ADR-2).
"""

from __future__ import annotations

import re

from secbrn.config import Settings
from secbrn.graph.base import GraphStore
from secbrn.models import ContextBundle, RetrievedChunk
from secbrn.providers.base import Embedder

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_NEIGHBOR_WEIGHT = 0.25  # far graph neighbours count far less than direct seed hits


def _terms(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) >= 3}


def _fuzzy_overlap(q: set[str], target: set[str]) -> int:
    """Count query terms that match a target term by equality/substring (>=3 chars)."""
    n = 0
    for qt in q:
        if any(qt == t or qt in t or t in qt for t in target):
            n += 1
    return n


def _rrf_fuse(ranked_lists: list[list[RetrievedChunk]], k: int, c: int = 60) -> list[RetrievedChunk]:
    """Reciprocal-rank fusion across multiple ranked lists, deduped by chunk_id."""
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    vias: dict[str, set[str]] = {}
    for lst in ranked_lists:
        for rank, rc in enumerate(lst):
            scores[rc.chunk_id] = scores.get(rc.chunk_id, 0.0) + 1.0 / (c + rank + 1)
            vias.setdefault(rc.chunk_id, set()).add(rc.via)
            if rc.chunk_id not in best or rc.score > best[rc.chunk_id].score:
                best[rc.chunk_id] = rc
    fused: list[RetrievedChunk] = []
    for cid, sc in sorted(scores.items(), key=lambda t: t[1], reverse=True)[:k]:
        rc = best[cid]
        rc.score = sc
        rc.via = "+".join(sorted(vias[cid]))
        fused.append(rc)
    return fused


class HybridRetriever:
    def __init__(self, settings: Settings, store: GraphStore, embedder: Embedder):
        self.s = settings
        self.store = store
        self.embedder = embedder

    # ── re-scoring ────────────────────────────────────────────────────────────────
    def _rescore(self, query: str, chunks: list[RetrievedChunk], hops: int) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        g_alpha = self.s.graph_boost
        t_alpha = self.s.title_boost
        qterms = _terms(query)

        # seeds = entities the query is literally about; neighbours = within `hops`.
        seeds: set[str] = set()
        neighbors: set[str] = set()
        if g_alpha > 0:
            seeds = set(self.store.match_entities(query, limit=self.s.retrieve_top_k))
            if seeds and hops > 0:
                _, nodes = self.store.expand(sorted(seeds), hops)
                neighbors = set(nodes) - seeds

        cmap = self.store.chunk_entity_map([c.chunk_id for c in chunks]) if (seeds or neighbors) else {}

        for c in chunks:
            factor = 1.0
            if g_alpha > 0 and (seeds or neighbors):
                ents = set(cmap.get(c.chunk_id, []))
                seed_hits = len(ents & seeds)
                neigh_hits = len(ents & neighbors)
                gboost = g_alpha * (seed_hits + _NEIGHBOR_WEIGHT * neigh_hits)
                if gboost:
                    factor += gboost
                    if "graph" not in c.via:
                        c.via = f"{c.via}+graph"
            if t_alpha > 0 and qterms:
                title_terms = _terms(c.document_title) | _terms(c.span.label or "")
                hits = _fuzzy_overlap(qterms, title_terms)
                if hits:
                    factor += t_alpha * (hits / len(qterms))
                    if "title" not in c.via:
                        c.via = f"{c.via}+title"
            c.score *= factor

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    # ── public API ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, *, top_k: int | None = None, hops: int | None = None) -> ContextBundle:
        top_k = top_k or self.s.retrieve_top_k
        hops = self.s.retrieve_hops if hops is None else hops

        qvec = self.embedder.embed_one(query)
        vector_hits = self.store.vector_search(qvec, top_k)
        fulltext_hits = self.store.fulltext_search(query, top_k)

        fused = _rrf_fuse([vector_hits, fulltext_hits], top_k)
        fused = self._rescore(query, fused, hops)

        # Graph expansion from entities mentioned in the fused chunks (for the subgraph).
        seed_entities = self.store.entities_for_chunks([rc.chunk_id for rc in fused])
        edges, entities = ([], seed_entities)
        if seed_entities and hops > 0:
            edges, entities = self.store.expand(seed_entities, hops)

        return ContextBundle(chunks=fused, edges=edges, entities=entities)

    def search(self, query: str, *, top_k: int | None = None, hops: int | None = None) -> list[RetrievedChunk]:
        """Hybrid chunk search with graph + title re-scoring (no subgraph assembly)."""
        top_k = top_k or self.s.retrieve_top_k
        hops = self.s.retrieve_hops if hops is None else hops
        qvec = self.embedder.embed_one(query)
        fused = _rrf_fuse(
            [self.store.vector_search(qvec, top_k), self.store.fulltext_search(query, top_k)],
            top_k,
        )
        return self._rescore(query, fused, hops)
