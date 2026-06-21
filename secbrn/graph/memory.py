"""In-memory GraphStore — same semantics as Neo4j, for offline runs and tests.

Not persistent. Cosine vector search and a naive token-overlap full-text scorer
mirror what the Neo4j vector + full-text indexes provide, so the read path can be
exercised without a database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from secbrn.graph.base import DocRef, GraphStore, StoredEntity, cosine
from secbrn.models import Chunk, Document, RetrievedChunk, Span, SubgraphEdge

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")


@dataclass
class _Doc:
    doc: Document


@dataclass
class _Ent:
    name: str
    label: str
    aliases: list[str] = field(default_factory=list)
    summary: str = ""
    mention_count: int = 0


class InMemoryStore(GraphStore):
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.has_chunk: dict[str, list[str]] = {}          # doc_id -> [chunk_id]
        self.entities: dict[str, _Ent] = {}                # name -> entity
        self.mentions: dict[str, set[str]] = {}            # chunk_id -> {entity_name}
        self.relations: set[tuple[str, str, str]] = set()  # (subj, rel, obj)
        self.doc_links: set[tuple[str, str]] = set()       # (doc_id, dst_uri)
        self.same_as: list[tuple[str, str]] = []           # history
        self._schema_ready = False

    # ── lifecycle ───────────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        self._schema_ready = True

    def close(self) -> None:
        pass

    def ping(self) -> bool:
        return True

    def indexes_present(self) -> dict[str, bool]:
        return {"chunk_vec": True, "chunk_ft": True, "entity_ft": True}

    # ── write path ───────────────────────────────────────────────────────────────
    def get_document_by_uri(self, uri: str) -> DocRef | None:
        for d in self.documents.values():
            if d.uri == uri:
                return DocRef(d.id, d.uri, d.content_hash, d.version)
        return None

    def content_hash_exists(self, content_hash: str) -> bool:
        return any(d.content_hash == content_hash for d in self.documents.values())

    def upsert_document(self, doc: Document) -> None:
        self.documents[doc.id] = doc
        self.has_chunk.setdefault(doc.id, [])

    def delete_chunks_for_document(self, document_id: str) -> None:
        for cid in self.has_chunk.get(document_id, []):
            self.chunks.pop(cid, None)
            self.mentions.pop(cid, None)
        self.has_chunk[document_id] = []

    def upsert_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.id] = chunk
        self.has_chunk.setdefault(chunk.document_id, [])
        if chunk.id not in self.has_chunk[chunk.document_id]:
            self.has_chunk[chunk.document_id].append(chunk.id)
        self.mentions.setdefault(chunk.id, set())

    def upsert_entity(self, name: str, label: str, aliases: list[str], summary: str = "") -> None:
        e = self.entities.get(name)
        if e is None:
            self.entities[name] = _Ent(name=name, label=label, aliases=list(aliases), summary=summary)
        else:
            for a in aliases:
                if a not in e.aliases:
                    e.aliases.append(a)
            if summary and not e.summary:
                e.summary = summary

    def add_mention(self, chunk_id: str, entity_name: str) -> None:
        self.mentions.setdefault(chunk_id, set()).add(entity_name)
        if entity_name in self.entities:
            self.entities[entity_name].mention_count += 1

    def add_relation(self, subject: str, relation: str, obj: str) -> None:
        self.relations.add((subject, relation, obj))

    def add_doc_link(self, src_document_id: str, dst_uri: str) -> None:
        self.doc_links.add((src_document_id, dst_uri))

    def add_authored_by(self, document_id: str, person_name: str) -> None:
        self.relations.add((document_id, "AUTHORED_BY", person_name))

    # ── resolution ───────────────────────────────────────────────────────────────
    def all_entities(self) -> list[StoredEntity]:
        return [
            StoredEntity(e.name, e.label, list(e.aliases), e.mention_count, e.summary)
            for e in self.entities.values()
        ]

    def entity_context_embedding(self, name: str) -> list[float] | None:
        vecs = [
            self.chunks[cid].embedding
            for cid, ents in self.mentions.items()
            if name in ents and self.chunks.get(cid) and self.chunks[cid].embedding
        ]
        if not vecs:
            return None
        dim = len(vecs[0])
        mean = [0.0] * dim
        for v in vecs:
            for i in range(dim):
                mean[i] += v[i]
        return [x / len(vecs) for x in mean]

    def merge_entities(self, canonical: str, duplicate: str, record_history: bool = True) -> None:
        if duplicate == canonical or duplicate not in self.entities:
            return
        canon = self.entities.setdefault(
            canonical, _Ent(name=canonical, label=self.entities[duplicate].label)
        )
        dup = self.entities.pop(duplicate)
        # merge aliases (keep the duplicate's surface form as an alias)
        for a in [duplicate, *dup.aliases]:
            if a != canonical and a not in canon.aliases:
                canon.aliases.append(a)
        canon.mention_count += dup.mention_count
        # rewire mentions
        for ents in self.mentions.values():
            if duplicate in ents:
                ents.discard(duplicate)
                ents.add(canonical)
        # rewire relations
        rewired = set()
        for s, r, o in self.relations:
            s2 = canonical if s == duplicate else s
            o2 = canonical if o == duplicate else o
            if s2 != o2:
                rewired.add((s2, r, o2))
        self.relations = rewired
        if record_history:
            self.same_as.append((duplicate, canonical))

    # ── read path ────────────────────────────────────────────────────────────────
    def _to_retrieved(self, chunk: Chunk, score: float, via: str) -> RetrievedChunk:
        doc = self.documents.get(chunk.document_id)
        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=doc.title if doc else chunk.document_id,
            uri=doc.uri if doc else "",
            text=chunk.text,
            span=chunk.span,
            score=score,
            via=via,
        )

    def vector_search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        scored = [
            (cosine(query_embedding, c.embedding), c)
            for c in self.chunks.values()
            if c.embedding
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [self._to_retrieved(c, s, "vector") for s, c in scored[:k]]

    def fulltext_search(self, query: str, k: int) -> list[RetrievedChunk]:
        q = set(_WORD.findall(query.lower()))
        results = []
        for c in self.chunks.values():
            toks = set(_WORD.findall(c.text.lower()))
            overlap = len(q & toks)
            if overlap:
                results.append((overlap / (len(q) or 1), c))
        results.sort(key=lambda t: t[0], reverse=True)
        return [self._to_retrieved(c, s, "fulltext") for s, c in results[:k]]

    def entities_for_chunks(self, chunk_ids: list[str]) -> list[str]:
        out: list[str] = []
        for cid in chunk_ids:
            for e in self.mentions.get(cid, set()):
                if e not in out:
                    out.append(e)
        return out

    def chunk_entity_map(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        return {cid: sorted(self.mentions.get(cid, set())) for cid in chunk_ids}

    def match_entities(self, query: str, limit: int = 10) -> list[str]:
        q = [t for t in _WORD.findall(query.lower()) if len(t) >= 3]
        scored: list[tuple[int, str]] = []
        for e in self.entities.values():
            toks = set()
            for f in (e.name, *e.aliases):
                toks |= set(_WORD.findall(f.lower()))
            # fuzzy: a query term matches an entity token by substring/prefix overlap,
            # so "graph" seeds "graphrag" / "Graph expansion".
            score = 0
            for qt in q:
                if any(qt == t or qt in t or t in qt for t in toks):
                    score += 1
            if score:
                scored.append((score, e.name))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [name for _, name in scored[:limit]]

    def expand(self, entity_names: list[str], hops: int) -> tuple[list[SubgraphEdge], list[str]]:
        frontier = set(entity_names)
        visited = set(entity_names)
        edges: list[SubgraphEdge] = []
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for s, r, o in self.relations:
                if s in frontier or o in frontier:
                    edges.append(SubgraphEdge(s, r, o))
                    for n in (s, o):
                        if n not in visited:
                            visited.add(n)
                            nxt.add(n)
            frontier = nxt
            if not frontier:
                break
        # dedup edges
        uniq = list({(e.subject, e.relation, e.object): e for e in edges}.values())
        return uniq, sorted(visited)

    # ── stats ────────────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, int]:
        orphan = sum(1 for cid in self.chunks if not self.mentions.get(cid))
        dup_alias = sum(len(e.aliases) for e in self.entities.values())
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "mentions": sum(len(v) for v in self.mentions.values()),
            "doc_links": len(self.doc_links),
            "orphan_chunks": orphan,
            "aliases": dup_alias,
            "merges": len(self.same_as),
        }
