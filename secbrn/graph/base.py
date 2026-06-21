"""GraphStore interface + small math helpers.

Two implementations conform to this:
  - :class:`secbrn.graph.neo4j_store.Neo4jStore` — the real backend.
  - :class:`secbrn.graph.memory.InMemoryStore` — offline/testing, same semantics.

Keeping the engine behind this interface is what lets neo4j-graphrag / LlamaIndex
plug in later (ARCHITECTURE ADR-2/3) without touching the pipeline.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from secbrn.models import Chunk, Document, RetrievedChunk, SubgraphEdge


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class StoredEntity:
    name: str
    label: str
    aliases: list[str]
    mention_count: int
    summary: str = ""


@dataclass
class DocRef:
    id: str
    uri: str
    content_hash: str
    version: int


class GraphStore(ABC):
    # ── lifecycle ──────────────────────────────────────────────────────────────
    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def ping(self) -> bool: ...

    @abstractmethod
    def indexes_present(self) -> dict[str, bool]:
        """Map of index name → present? Used by healthcheck."""

    # ── write path ─────────────────────────────────────────────────────────────
    @abstractmethod
    def get_document_by_uri(self, uri: str) -> DocRef | None: ...

    @abstractmethod
    def content_hash_exists(self, content_hash: str) -> bool: ...

    @abstractmethod
    def upsert_document(self, doc: Document) -> None: ...

    @abstractmethod
    def delete_chunks_for_document(self, document_id: str) -> None:
        """Remove a document's chunks + their MENTIONS (for re-extraction on update)."""

    @abstractmethod
    def upsert_chunk(self, chunk: Chunk) -> None: ...

    @abstractmethod
    def upsert_entity(self, name: str, label: str, aliases: list[str], summary: str = "") -> None: ...

    @abstractmethod
    def add_mention(self, chunk_id: str, entity_name: str) -> None: ...

    @abstractmethod
    def add_relation(self, subject: str, relation: str, obj: str) -> None: ...

    @abstractmethod
    def add_doc_link(self, src_document_id: str, dst_uri: str) -> None: ...

    @abstractmethod
    def add_authored_by(self, document_id: str, person_name: str) -> None: ...

    # ── resolution (Stage 6) ───────────────────────────────────────────────────
    @abstractmethod
    def all_entities(self) -> list[StoredEntity]: ...

    @abstractmethod
    def entity_context_embedding(self, name: str) -> list[float] | None:
        """Mean embedding of chunks that mention the entity (for similarity scoring)."""

    @abstractmethod
    def merge_entities(self, canonical: str, duplicate: str, record_history: bool = True) -> None:
        """Collapse ``duplicate`` into ``canonical``: rewire edges, merge aliases."""

    # ── read path (Stage 7) ─────────────────────────────────────────────────────
    @abstractmethod
    def vector_search(self, query_embedding: list[float], k: int) -> list[RetrievedChunk]: ...

    @abstractmethod
    def fulltext_search(self, query: str, k: int) -> list[RetrievedChunk]: ...

    @abstractmethod
    def entities_for_chunks(self, chunk_ids: list[str]) -> list[str]: ...

    @abstractmethod
    def chunk_entity_map(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        """Map each chunk id -> the entity names it mentions (for graph-aware scoring)."""

    @abstractmethod
    def match_entities(self, query: str, limit: int = 10) -> list[str]:
        """Entity names whose name/alias matches terms in the query (seed entities)."""

    @abstractmethod
    def expand(self, entity_names: list[str], hops: int) -> tuple[list[SubgraphEdge], list[str]]:
        """Traverse typed relations up to ``hops`` from the given entities."""

    # ── stats ──────────────────────────────────────────────────────────────────
    @abstractmethod
    def stats(self) -> dict[str, int]: ...
