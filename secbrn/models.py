"""Core data shapes shared across the pipeline.

Two structural node types (created deterministically, not LLM-extracted) plus the
extraction result types. See docs/SCHEMA.md §1.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    """Stable content hash used for idempotent ingestion (Stage 2)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Span:
    """A precise back-pointer into a source, for citations.

    ``kind`` is one of: ``page`` (PDF), ``line`` (markdown/text), ``turn``
    (transcript), ``section`` (web). ``start``/``end`` are interpreted per-kind.
    """

    kind: str
    start: int
    end: int
    label: str | None = None  # e.g. heading text, speaker, section title

    def cite(self) -> str:
        base = f"{self.kind} {self.start}" + (f"–{self.end}" if self.end != self.start else "")
        return f"{base} ({self.label})" if self.label else base


@dataclass
class Document:
    """A normalized source. Stage 1 produces a raw one; Stage 2 finalizes hash/version."""

    id: str
    source_type: str            # markdown | pdf | web | transcript
    uri: str
    title: str
    raw_text: str
    created_at: str = field(default_factory=utcnow_iso)
    content_hash: str = ""
    version: int = 1
    schema_version: int = 1
    # explicit links discovered at load time (wikilinks / hyperlinks) → Stage 5 promotion
    links: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def finalize_hash(self) -> "Document":
        if not self.content_hash:
            self.content_hash = content_hash(self.raw_text)
        return self


@dataclass
class Chunk:
    """A passage of a Document. Holds text + embedding; vector-indexed in Neo4j."""

    id: str
    document_id: str
    position: int
    text: str
    span: Span
    embedding: list[float] | None = None
    embed_model: str | None = None
    embed_dim: int | None = None


@dataclass
class ExtractedEntity:
    name: str
    label: str                  # Concept | Tool | Person | Org | Topic | Event | Place
    aliases: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class ExtractedRelation:
    subject: str                # entity name
    relation: str               # RELATES_TO | PART_OF | USES | ...
    object: str                 # entity name


@dataclass
class Extraction:
    """Result of Stage 5 for one chunk."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


# ── Retrieval / answering shapes (read path) ───────────────────────────────────


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    uri: str
    text: str
    span: Span
    score: float
    via: str                    # "vector" | "fulltext" | "graph"


@dataclass
class SubgraphEdge:
    subject: str
    relation: str
    object: str


@dataclass
class ContextBundle:
    """Stage 7 output: the fused, deduplicated context handed to synthesis."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    edges: list[SubgraphEdge] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass
class Citation:
    marker: str                 # e.g. "[1]"
    document_title: str
    uri: str
    span: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    bundle: ContextBundle | None = None
    uncited: bool = False
