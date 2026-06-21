"""The engine: :class:`Brain` wires Stages 1-8 together.

This is the importable Python API; the CLI (and later the web UI / MCP server) are
thin wrappers over it.

    brain = Brain.from_env()
    brain.ingest("./notes/")
    print(brain.ask("How does retrieval relate to rerankers?").text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from secbrn.answer import synthesize
from secbrn.config import Settings, get_settings
from secbrn.extract import extract_chunk
from secbrn.graph import GraphStore, get_store
from secbrn.ingest import (
    chunk_document,
    iter_folder,
    load_path,
    load_web,
    normalize_and_dedupe,
)
from secbrn.models import Answer, Document, RetrievedChunk
from secbrn.providers import get_answer_llm, get_embedder, get_extract_llm
from secbrn.providers.base import Embedder, LLM
from secbrn.resolve import EntityResolver, MergeDecision
from secbrn.retrieve import HybridRetriever


@dataclass
class IngestReport:
    documents_ingested: int = 0
    documents_skipped: int = 0
    documents_updated: int = 0
    chunks_written: int = 0
    chunks_failed: int = 0          # embedding failed (e.g. timeout) -> chunk skipped
    extractions_failed: int = 0     # KG extraction failed -> chunk kept, no entities
    entities_extracted: int = 0
    relations_extracted: int = 0
    merges: int = 0
    details: list[str] = field(default_factory=list)


class Brain:
    def __init__(
        self,
        settings: Settings,
        store: GraphStore,
        embedder: Embedder,
        extract_llm: LLM,
        answer_llm: LLM,
    ):
        self.s = settings
        self.store = store
        self.embedder = embedder
        self.extract_llm = extract_llm
        self.answer_llm = answer_llm
        self.retriever = HybridRetriever(settings, store, embedder)
        self.store.ensure_schema()

    # -- construction -----------------------------------------------------------------
    @classmethod
    def from_env(cls, settings: Settings | None = None) -> "Brain":
        s = settings or get_settings()
        return cls(
            settings=s,
            store=get_store(s),
            embedder=get_embedder(s),
            extract_llm=get_extract_llm(s),
            answer_llm=get_answer_llm(s),
        )

    @classmethod
    def isolated(cls, settings: Settings | None = None) -> "Brain":
        """A throwaway in-memory brain using the configured (real) providers.

        Used to evaluate retrieval against ONLY a gold corpus, so unrelated data in the
        main Neo4j store can't dilute the metrics.
        """
        s = settings or get_settings()
        from secbrn.graph.memory import InMemoryStore

        return cls(
            settings=s,
            store=InMemoryStore(),
            embedder=get_embedder(s),
            extract_llm=get_extract_llm(s),
            answer_llm=get_answer_llm(s),
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Brain":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write path (Stages 1-6) ------------------------------------------------------
    def ingest(self, path: str | Path, *, resolve: bool = True, raise_errors: bool = False) -> IngestReport:
        """Ingest a file or folder. Idempotent: unchanged files are skipped.

        Resilient by design: a bad *file* is logged to ``report.details`` and ingestion
        continues; a bad *chunk* (e.g. an Ollama timeout) is skipped without discarding
        the rest of the document. Pass ``raise_errors=True`` (CLI ``--debug``) to
        re-raise the first failure with a full traceback.
        """
        p = Path(path)
        files = iter_folder(p) if p.is_dir() else [p]
        report = IngestReport()
        for f in files:
            try:
                self._ingest_document(load_path(f), report, raise_errors=raise_errors)
            except Exception as e:  # keep going on a bad file
                if raise_errors:
                    raise
                report.details.append(f"ERROR {f}: {type(e).__name__}: {e}")
        if resolve:
            report.merges = len(self.resolve())
        return report

    def ingest_url(self, url: str, *, html: str | None = None, resolve: bool = True,
                   raise_errors: bool = False) -> IngestReport:
        report = IngestReport()
        self._ingest_document(load_web(url, html=html), report, raise_errors=raise_errors)
        if resolve:
            report.merges = len(self.resolve())
        return report

    def _ingest_document(self, doc: Document, report: IngestReport, *, raise_errors: bool = False) -> None:
        # Stage 2 -- normalize + dedupe/version
        result = normalize_and_dedupe(doc, self.store)
        if result.status == "duplicate":
            report.documents_skipped += 1
            report.details.append(f"skip (unchanged): {doc.title}")
            return
        doc = result.document
        if result.status == "updated":
            self.store.delete_chunks_for_document(doc.id)  # re-extract, don't duplicate
            report.documents_updated += 1
        else:
            report.documents_ingested += 1

        self.store.upsert_document(doc)

        # Stage 3 -- chunk
        chunks = chunk_document(doc, chunk_size=self.s.chunk_size, overlap=self.s.chunk_overlap)

        # Stages 4 + 5 -- embed/write then extract, PER CHUNK so one failure (e.g. a
        # ReadTimeout on a huge PDF) skips just that chunk instead of aborting the doc.
        written: list = []
        for c in chunks:
            try:
                c.embedding = self.embedder.embed_one(c.text)
                c.embed_model = self.embedder.model
                c.embed_dim = self.embedder.dim
                self.store.upsert_chunk(c)
                written.append(c)
                report.chunks_written += 1
            except Exception as e:
                if raise_errors:
                    raise
                report.chunks_failed += 1
                report.details.append(f"embed failed [{doc.title} chunk {c.position}]: {type(e).__name__}: {e}")

        for c in written:
            try:
                ex = extract_chunk(c.text, self.extract_llm)
            except Exception as e:
                if raise_errors:
                    raise
                report.extractions_failed += 1
                report.details.append(f"extract failed [{doc.title} chunk {c.position}]: {type(e).__name__}: {e}")
                continue
            for ent in ex.entities:
                self.store.upsert_entity(ent.name, ent.label, ent.aliases, ent.summary)
                self.store.add_mention(c.id, ent.name)
                report.entities_extracted += 1
            for rel in ex.relations:
                self.store.add_relation(rel.subject, rel.relation, rel.object)
                report.relations_extracted += 1

        # explicit-link promotion (wikilinks / hyperlinks -> LINKS_TO)
        for link in doc.links:
            self.store.add_doc_link(doc.id, link)

    # -- Stage 6 on demand ------------------------------------------------------------
    def resolve(self) -> list[MergeDecision]:
        resolver = EntityResolver(self.s, self.store, llm=self.extract_llm)
        return resolver.run()

    # -- read path (Stages 7-8) -------------------------------------------------------
    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.search(query, top_k=top_k)

    def ask(self, question: str, *, top_k: int | None = None, hops: int | None = None) -> Answer:
        bundle = self.retriever.retrieve(question, top_k=top_k, hops=hops)
        return synthesize(question, bundle, self.answer_llm)

    # -- ops --------------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        return self.store.stats()

    def reindex(self, *, recreate: bool = False) -> None:
        """Re-apply DDL (idempotent). With recreate=True, drop+recreate the vector index
        at the configured dimension (needed after switching embedding models)."""
        self.store.ensure_schema()
        if recreate:
            self.store.recreate_vector_index()
