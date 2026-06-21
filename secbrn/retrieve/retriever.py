"""Stage 7 — Hybrid retrieval: vector + full-text + graph + lexical, with optional
query expansion (pre-retrieval) and LLM reranking (post-fusion).

Per query:
  0. (optional) expand the query with LLM keywords for vector + full-text search.
  1. Vector search over chunk embeddings.
  2. Full-text / keyword search over chunk text.
  3. Reciprocal-rank fusion of (1) and (2).
  4. Re-score: seed-weighted graph boost + title/heading lexical boost.
  5. (optional) listwise LLM rerank of the top candidates.
  6. Graph expansion from the top chunks' entities builds the answer's subgraph.

The original query (not the expanded one) drives entity seeding and title matching, so
expansion can't pollute those exact-term signals.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from secbrn.config import Settings
from secbrn.graph.base import GraphStore
from secbrn.models import ContextBundle, RetrievedChunk
from secbrn.providers.base import Embedder, LLM
from secbrn.retrieve.expand import expand_query
from secbrn.retrieve.rerank import LLMReranker

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_NEIGHBOR_WEIGHT = 0.25  # far graph neighbours count far less than direct seed hits


def _terms(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) >= 3}


def _fuzzy_overlap(q: set[str], target: set[str]) -> int:
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
    def __init__(self, settings: Settings, store: GraphStore, embedder: Embedder,
                 llm: LLM | None = None, reranker: LLMReranker | None = None):
        self.s = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker

    # ── helpers ──────────────────────────────────────────────────────────────────
    def _search_query(self, query: str) -> str:
        if self.s.query_expansion and self.llm is not None:
            return expand_query(query, self.llm, self.s.query_expansion_terms)
        return query

    def _fuse(self, search_query: str, top_k: int) -> list[RetrievedChunk]:
        # Full-text search doesn't need the query embedding, so run it in a background
        # thread while we spend the (usually dominant) wall-clock on the embed round-trip
        # plus the vector query. The two store reads then overlap instead of serializing.
        with ThreadPoolExecutor(max_workers=1) as pool:
            ft_future = pool.submit(self.store.fulltext_search, search_query, top_k)
            qvec = self.embedder.embed_one(search_query)
            vector_hits = self.store.vector_search(qvec, top_k)
            fulltext_hits = ft_future.result()
        return _rrf_fuse([vector_hits, fulltext_hits], top_k)

    def _rescore(self, query: str, chunks: list[RetrievedChunk], hops: int) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        g_alpha = self.s.graph_boost
        t_alpha = self.s.title_boost
        qterms = _terms(query)

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
                gboost = g_alpha * (len(ents & seeds) + _NEIGHBOR_WEIGHT * len(ents & neighbors))
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

    def _maybe_rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.s.rerank and self.reranker is not None and len(chunks) > 1:
            return self.reranker.rerank(query, chunks, self.s.rerank_candidates)
        return chunks

    # ── public API ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, *, top_k: int | None = None, hops: int | None = None) -> ContextBundle:
        top_k = top_k or self.s.retrieve_top_k
        hops = self.s.retrieve_hops if hops is None else hops

        fused = self._fuse(self._search_query(query), top_k)
        fused = self._rescore(query, fused, hops)
        fused = self._maybe_rerank(query, fused)

        seed_entities = self.store.entities_for_chunks([rc.chunk_id for rc in fused])
        edges, entities = ([], seed_entities)
        if seed_entities and hops > 0:
            edges, entities = self.store.expand(seed_entities, hops)
        return ContextBundle(chunks=fused, edges=edges, entities=entities)

    def search(self, query: str, *, top_k: int | None = None, hops: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.s.retrieve_top_k
        hops = self.s.retrieve_hops if hops is None else hops
        fused = self._fuse(self._search_query(query), top_k)
        fused = self._rescore(query, fused, hops)
        return self._maybe_rerank(query, fused)
