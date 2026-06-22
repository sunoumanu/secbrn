"""Neo4j-backed GraphStore — the real engine. Uses the official driver + Cypher.

Vector + full-text retrieval use Neo4j's native indexes
(``db.index.vector.queryNodes`` / ``db.index.fulltext.queryNodes``).
"""

from __future__ import annotations

import json

from secbrn.graph.base import DocRef, GraphStore, StoredEntity
from secbrn.graph.ddl import all_ddl, vector_index_ddl
from secbrn.models import Chunk, Document, RetrievedChunk, Span, SubgraphEdge

try:  # neo4j is an install dep, but keep import soft so `fake`/in-memory works without it
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore


def _span_to_json(span: Span) -> str:
    return json.dumps({"kind": span.kind, "start": span.start, "end": span.end, "label": span.label})


def _span_from_json(s: str | None) -> Span:
    if not s:
        return Span("line", 0, 0)
    d = json.loads(s)
    return Span(d.get("kind", "line"), d.get("start", 0), d.get("end", 0), d.get("label"))


class Neo4jStore(GraphStore):
    def __init__(self, uri: str, user: str, password: str, database: str, embed_dim: int):
        if GraphDatabase is None:  # pragma: no cover
            raise RuntimeError("neo4j driver not installed. `pip install neo4j`.")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._embed_dim = embed_dim

    # ── helpers ────────────────────────────────────────────────────────────────
    def _run(self, cypher: str, **params):
        with self._driver.session(database=self._db) as session:
            return list(session.run(cypher, **params))

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        # Self-heal a stale vector index. The DDL creates chunk_vec with
        # ``IF NOT EXISTS``, so if an index already exists at a different dimension
        # (e.g. you switched embedding models 768 → 1024) it would be silently kept
        # and vector_search would later fail with a dim-mismatch ClientError. Detect
        # that here and drop the old index so the DDL below recreates it correctly.
        existing_dim = self.vector_index_dim()
        if existing_dim is not None and existing_dim != self._embed_dim:
            self._run("DROP INDEX chunk_vec IF EXISTS")
        for stmt in all_ddl(self._embed_dim):
            self._run(stmt)

    def vector_index_dim(self) -> int | None:
        """Dimension the ``chunk_vec`` vector index was created with, or None if absent."""
        rows = self._run(
            "SHOW INDEXES YIELD name, options "
            "WHERE name = 'chunk_vec' RETURN options AS options"
        )
        if not rows:
            return None
        opts = rows[0]["options"] or {}
        cfg = opts.get("indexConfig") or {}
        dim = cfg.get("vector.dimensions")
        return int(dim) if dim is not None else None

    def recreate_vector_index(self) -> None:
        """Drop and recreate the chunk vector index at the configured dimension.

        Needed when switching to an embedding model with a different dimension.
        """
        self._run("DROP INDEX chunk_vec IF EXISTS")
        self._run(vector_index_ddl(self._embed_dim))

    def sample_chunk_embed_dim(self) -> int | None:
        rows = self._run(
            "MATCH (c:Chunk) WHERE c.embed_dim IS NOT NULL RETURN c.embed_dim AS d LIMIT 1"
        )
        return rows[0]["d"] if rows else None

    def close(self) -> None:
        self._driver.close()

    def clear(self) -> None:
        """Delete every node (and its relationships); keep constraints + indexes.

        Batched via ``CALL (n) { … } IN TRANSACTIONS`` so a large graph doesn't blow the
        heap in one transaction. Runs in auto-commit mode (required by IN TRANSACTIONS).
        """
        self._run(
            "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
        )

    def ping(self) -> bool:
        try:
            self._run("RETURN 1 AS ok")
            return True
        except Exception:
            return False

    def indexes_present(self) -> dict[str, bool]:
        rows = self._run("SHOW INDEXES YIELD name RETURN collect(name) AS names")
        names = set(rows[0]["names"]) if rows else set()
        return {n: (n in names) for n in ("chunk_vec", "chunk_ft", "entity_ft")}

    # ── write path ────────────────────────────────────────────────────────────────
    def get_document_by_uri(self, uri: str) -> DocRef | None:
        rows = self._run(
            "MATCH (d:Document {uri:$uri}) "
            "RETURN d.id AS id, d.uri AS uri, d.content_hash AS h, d.version AS v",
            uri=uri,
        )
        if not rows:
            return None
        r = rows[0]
        return DocRef(r["id"], r["uri"], r["h"], r["v"])

    def content_hash_exists(self, content_hash: str) -> bool:
        rows = self._run(
            "MATCH (d:Document {content_hash:$h}) RETURN count(d) AS n", h=content_hash
        )
        return rows[0]["n"] > 0

    def upsert_document(self, doc: Document) -> None:
        self._run(
            """
            MERGE (d:Document {id:$id})
            SET d.source_type=$source_type, d.uri=$uri, d.title=$title,
                d.created_at=$created_at, d.content_hash=$content_hash,
                d.version=$version, d.schema_version=$schema_version
            """,
            id=doc.id, source_type=doc.source_type, uri=doc.uri, title=doc.title,
            created_at=doc.created_at, content_hash=doc.content_hash,
            version=doc.version, schema_version=doc.schema_version,
        )

    def delete_chunks_for_document(self, document_id: str) -> None:
        self._run(
            "MATCH (d:Document {id:$id})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c",
            id=document_id,
        )

    def upsert_chunk(self, chunk: Chunk) -> None:
        self._run(
            """
            MATCH (d:Document {id:$doc})
            MERGE (c:Chunk {id:$id})
            SET c.text=$text, c.position=$position, c.span=$span,
                c.embed_model=$embed_model, c.embed_dim=$embed_dim
            MERGE (d)-[:HAS_CHUNK]->(c)
            WITH c
            CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
            RETURN c.id
            """,
            doc=chunk.document_id, id=chunk.id, text=chunk.text, position=chunk.position,
            span=_span_to_json(chunk.span), embed_model=chunk.embed_model,
            embed_dim=chunk.embed_dim, embedding=chunk.embedding,
        )

    def update_chunk_span(self, chunk_id: str, span: Span) -> bool:
        """Rewrite just a chunk's citation span in place (no embed/extract touched).

        Used by the span backfill to repair chunks ingested before line spans were
        computed correctly. Returns True if a chunk with that id existed.
        """
        rows = self._run(
            "MATCH (c:Chunk {id:$id}) SET c.span=$span RETURN count(c) AS n",
            id=chunk_id, span=_span_to_json(span),
        )
        return bool(rows and rows[0]["n"])

    def upsert_entity(self, name: str, label: str, aliases: list[str], summary: str = "") -> None:
        # Apply both :Entity and the semantic label. Label is parameterised via APOC-free
        # dynamic label set using a CALL — but to avoid APOC dependency we whitelist labels.
        self._run(
            f"""
            MERGE (e:Entity {{name:$name}})
            ON CREATE SET e.created_at=datetime(), e.mention_count=0, e.aliases=$aliases,
                          e.summary=$summary
            SET e:{label}
            SET e.aliases = apoc.coll.toSet(coalesce(e.aliases,[]) + $aliases)
            """ if _has_apoc_safe() else
            f"""
            MERGE (e:Entity {{name:$name}})
            ON CREATE SET e.created_at=datetime(), e.mention_count=0, e.summary=$summary
            SET e:{label}
            SET e.aliases = [a IN coalesce(e.aliases, []) WHERE NOT a IN $aliases] + $aliases
            """,
            name=name, aliases=aliases, summary=summary,
        )

    def add_mention(self, chunk_id: str, entity_name: str) -> None:
        self._run(
            """
            MATCH (c:Chunk {id:$cid}) MATCH (e:Entity {name:$name})
            MERGE (c)-[:MENTIONS]->(e)
            SET e.mention_count = coalesce(e.mention_count,0) + 1
            """,
            cid=chunk_id, name=entity_name,
        )

    def add_relation(self, subject: str, relation: str, obj: str) -> None:
        # relation type is whitelisted by the schema before reaching here
        self._run(
            f"""
            MATCH (a:Entity {{name:$s}}) MATCH (b:Entity {{name:$o}})
            MERGE (a)-[:{relation}]->(b)
            """,
            s=subject, o=obj,
        )

    def add_doc_link(self, src_document_id: str, dst_uri: str) -> None:
        self._run(
            """
            MATCH (a:Document {id:$src})
            MATCH (b:Document {uri:$dst})
            MERGE (a)-[:LINKS_TO]->(b)
            """,
            src=src_document_id, dst=dst_uri,
        )

    def add_authored_by(self, document_id: str, person_name: str) -> None:
        self._run(
            """
            MATCH (d:Document {id:$id})
            MERGE (p:Entity {name:$name}) SET p:Person
            MERGE (d)-[:AUTHORED_BY]->(p)
            """,
            id=document_id, name=person_name,
        )

    # ── resolution ───────────────────────────────────────────────────────────────
    def all_entities(self) -> list[StoredEntity]:
        rows = self._run(
            """
            MATCH (e:Entity)
            RETURN e.name AS name,
                   [l IN labels(e) WHERE l <> 'Entity'][0] AS label,
                   coalesce(e.aliases, []) AS aliases,
                   coalesce(e.mention_count, 0) AS mc,
                   coalesce(e.summary, '') AS summary
            """
        )
        return [
            StoredEntity(r["name"], r["label"] or "Concept", r["aliases"], r["mc"], r["summary"])
            for r in rows
        ]

    def entity_context_embedding(self, name: str) -> list[float] | None:
        rows = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(:Entity {name:$name})
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding AS emb
            """,
            name=name,
        )
        if not rows:
            return None
        vecs = [r["emb"] for r in rows]
        dim = len(vecs[0])
        mean = [0.0] * dim
        for v in vecs:
            for i in range(dim):
                mean[i] += v[i]
        return [x / len(vecs) for x in mean]

    def merge_entities(self, canonical: str, duplicate: str, record_history: bool = True) -> None:
        if duplicate == canonical:
            return
        # Rewire relationships and mentions onto the canonical node, then merge alias
        # lists and delete the duplicate. Uses APOC if available for clean edge rewiring;
        # falls back to manual rewire otherwise.
        self._run(
            """
            MATCH (dup:Entity {name:$dup})
            MATCH (canon:Entity {name:$canon})
            // move incoming MENTIONS
            CALL (dup, canon) {
              MATCH (c)-[m:MENTIONS]->(dup)
              MERGE (c)-[:MENTIONS]->(canon)
              DELETE m
            }
            // move outgoing/incoming entity relations generically via apoc if present
            WITH dup, canon
            SET canon.aliases = [a IN coalesce(canon.aliases,[])
                                 WHERE a <> $dup AND NOT a IN coalesce(dup.aliases,[])]
                                + [$dup] + coalesce(dup.aliases,[]),
                canon.mention_count = coalesce(canon.mention_count,0)
                                      + coalesce(dup.mention_count,0)
            """,
            dup=duplicate, canon=canonical,
        )
        # Rewire typed entity-entity relations (manual, APOC-free).
        for rel in ("RELATES_TO", "PART_OF", "USES", "IMPROVES", "ALTERNATIVE_TO"):
            self._run(
                f"""
                MATCH (dup:Entity {{name:$dup}})-[r:{rel}]->(x)
                MATCH (canon:Entity {{name:$canon}})
                WHERE x.name <> $canon
                MERGE (canon)-[:{rel}]->(x) DELETE r
                """,
                dup=duplicate, canon=canonical,
            )
            self._run(
                f"""
                MATCH (x)-[r:{rel}]->(dup:Entity {{name:$dup}})
                MATCH (canon:Entity {{name:$canon}})
                WHERE x.name <> $canon
                MERGE (x)-[:{rel}]->(canon) DELETE r
                """,
                dup=duplicate, canon=canonical,
            )
        if record_history:
            self._run(
                """
                MATCH (canon:Entity {name:$canon})
                MERGE (h:MergeHistory {duplicate:$dup, canonical:$canon, at:toString(datetime())})
                MERGE (canon)-[:SAME_AS]->(h)
                """,
                dup=duplicate, canon=canonical,
            )
        self._run("MATCH (dup:Entity {name:$dup}) DETACH DELETE dup", dup=duplicate)

    # ── read path ────────────────────────────────────────────────────────────────
    def _row_to_retrieved(self, r, via: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=r["cid"],
            document_id=r["did"],
            document_title=r["title"] or r["did"],
            uri=r["uri"] or "",
            text=r["text"],
            span=_span_from_json(r["span"]),
            score=float(r["score"]),
            via=via,
        )

    def vector_search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]:
        rows = self._run(
            """
            CALL db.index.vector.queryNodes('chunk_vec', $k, $emb)
            YIELD node AS c, score
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            RETURN c.id AS cid, d.id AS did, d.title AS title, d.uri AS uri,
                   c.text AS text, c.span AS span, score
            """,
            k=k, emb=query_embedding,
        )
        return [self._row_to_retrieved(r, "vector") for r in rows]

    def fulltext_search(self, query: str, k: int) -> list[RetrievedChunk]:
        rows = self._run(
            """
            CALL db.index.fulltext.queryNodes('chunk_ft', $q) YIELD node AS c, score
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            RETURN c.id AS cid, d.id AS did, d.title AS title, d.uri AS uri,
                   c.text AS text, c.span AS span, score
            LIMIT $k
            """,
            q=_lucene_escape(query), k=k,
        )
        return [self._row_to_retrieved(r, "fulltext") for r in rows]

    def entities_for_chunks(self, chunk_ids: list[str]) -> list[str]:
        rows = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN $ids
            RETURN collect(DISTINCT e.name) AS names
            """,
            ids=chunk_ids,
        )
        return rows[0]["names"] if rows else []

    def chunk_entity_map(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        rows = self._run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE c.id IN $ids
            RETURN c.id AS cid, collect(DISTINCT e.name) AS names
            """,
            ids=chunk_ids,
        )
        return {r["cid"]: r["names"] for r in rows}

    def match_entities(self, query: str, limit: int = 10) -> list[str]:
        # Build a fuzzy OR query: each term as a prefix wildcard, so "graph" also seeds
        # "GraphRAG" / "Graph expansion". Falls back to escaped phrase if no terms.
        import re as _re

        terms = [t for t in _re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query) if len(t) >= 3]
        q = " OR ".join(f"{_lucene_escape(t)}*" for t in terms) or _lucene_escape(query)
        rows = self._run(
            """
            CALL db.index.fulltext.queryNodes('entity_ft', $q) YIELD node AS e, score
            RETURN e.name AS name ORDER BY score DESC LIMIT $limit
            """,
            q=q, limit=limit,
        )
        return [r["name"] for r in rows]

    def expand(self, entity_names: list[str], hops: int) -> tuple[list[SubgraphEdge], list[str]]:
        hops = max(1, hops)
        rows = self._run(
            f"""
            MATCH (a:Entity) WHERE a.name IN $names
            MATCH p = (a)-[r*1..{hops}]-(b:Entity)
            WITH relationships(p) AS rels
            UNWIND rels AS rel
            WITH startNode(rel) AS s, type(rel) AS t, endNode(rel) AS o
            WHERE 'Entity' IN labels(s) AND 'Entity' IN labels(o)
            RETURN DISTINCT s.name AS subj, t AS rel, o.name AS obj
            """,
            names=entity_names,
        )
        edges = [SubgraphEdge(r["subj"], r["rel"], r["obj"]) for r in rows]
        nodes = set(entity_names)
        for e in edges:
            nodes.add(e.subject)
            nodes.add(e.object)
        return edges, sorted(nodes)

    # ── stats ────────────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, int]:
        def n(cypher: str) -> int:
            rows = self._run(cypher)
            return rows[0]["n"] if rows else 0

        return {
            "documents": n("MATCH (d:Document) RETURN count(d) AS n"),
            "chunks": n("MATCH (c:Chunk) RETURN count(c) AS n"),
            "entities": n("MATCH (e:Entity) RETURN count(e) AS n"),
            "relations": n(
                "MATCH (:Entity)-[r]->(:Entity) "
                "WHERE type(r) IN ['RELATES_TO','PART_OF','USES','IMPROVES','ALTERNATIVE_TO'] "
                "RETURN count(r) AS n"
            ),
            "mentions": n("MATCH (:Chunk)-[m:MENTIONS]->(:Entity) RETURN count(m) AS n"),
            "doc_links": n("MATCH (:Document)-[l:LINKS_TO]->(:Document) RETURN count(l) AS n"),
            "orphan_chunks": n(
                "MATCH (c:Chunk) WHERE NOT (c)-[:MENTIONS]->() RETURN count(c) AS n"
            ),
            "merges": n("MATCH (:Entity)-[:SAME_AS]->(:MergeHistory) RETURN count(*) AS n"),
        }


def _lucene_escape(q: str) -> str:
    specials = r'+-&|!(){}[]^"~*?:\/'
    out = []
    for ch in q:
        if ch in specials:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out) or "*"


def _has_apoc_safe() -> bool:
    # Conservative: don't assume APOC. The fallback Cypher is APOC-free.
    return False
