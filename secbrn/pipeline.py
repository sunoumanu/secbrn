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
from typing import Callable

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
from secbrn.providers import get_answer_llm, get_embedder, get_extract_llm, get_rerank_llm
from secbrn.providers.base import Embedder, LLM
from secbrn.resolve import EntityResolver, MergeDecision
from secbrn.retrieve import HybridRetriever
from secbrn.retrieve.rerank import LLMReranker
from secbrn.util import map_workers


@dataclass
class IngestProgress:
    """A single progress tick emitted during ingest, for a CLI/UI to render.

    ``phase`` is "embed" or "extract"; ``done``/``total`` count chunks within the
    current document's current phase. A tick with ``done == 0`` marks phase start
    (so a UI can create/size a bar before any work completes).
    """

    phase: str
    doc_title: str
    done: int
    total: int


# A progress sink: called with each IngestProgress event. Kept as a plain callable so
# the engine stays UI-agnostic (the CLI wires this to a rich progress bar).
ProgressFn = Callable[[IngestProgress], None]


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
        rerank_llm = get_rerank_llm(settings)
        reranker = LLMReranker(rerank_llm) if settings.rerank else None
        self.retriever = HybridRetriever(settings, store, embedder, llm=rerank_llm, reranker=reranker)
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
    def ingest(self, path: str | Path, *, resolve: bool = True, raise_errors: bool = False,
               progress: ProgressFn | None = None) -> IngestReport:
        """Ingest a file or folder. Idempotent: unchanged files are skipped.

        Resilient by design: a bad *file* is logged to ``report.details`` and ingestion
        continues; a bad *chunk* (e.g. an Ollama timeout) is skipped without discarding
        the rest of the document. Pass ``raise_errors=True`` (CLI ``--debug``) to
        re-raise the first failure with a full traceback. ``progress`` (optional)
        receives :class:`IngestProgress` ticks for a live progress display.
        """
        p = Path(path)
        files = iter_folder(p) if p.is_dir() else [p]
        report = IngestReport()
        for f in files:
            try:
                self._ingest_document(load_path(f), report, raise_errors=raise_errors,
                                      progress=progress)
            except Exception as e:  # keep going on a bad file
                if raise_errors:
                    raise
                report.details.append(f"ERROR {f}: {type(e).__name__}: {e}")
        if resolve:
            report.merges = len(self.resolve())
        return report

    def ingest_url(self, url: str, *, html: str | None = None, resolve: bool = True,
                   raise_errors: bool = False, progress: ProgressFn | None = None) -> IngestReport:
        report = IngestReport()
        self._ingest_document(load_web(url, html=html), report, raise_errors=raise_errors,
                              progress=progress)
        if resolve:
            report.merges = len(self.resolve())
        return report

    def _ingest_document(self, doc: Document, report: IngestReport, *, raise_errors: bool = False,
                         progress: ProgressFn | None = None) -> None:
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

        # Stages 4 + 5 -- embed then extract, PER CHUNK so one failure (e.g. a ReadTimeout
        # on a huge PDF) skips just that chunk instead of aborting the doc.
        #
        # The embed and extract calls are blocking Ollama round-trips and are the dominant
        # ingest cost, so we run them with bounded concurrency (``ingest_concurrency``).
        # Only the *network* work is parallel: every store write stays on this thread, in
        # chunk order, so resilience reporting and Neo4j access remain serial and
        # deterministic. ``map_workers`` returns one (result, exc) per chunk IN ORDER, so
        # the previous per-chunk try/except semantics are preserved exactly.
        workers = self.s.ingest_concurrency

        # Stage 4 -- embed (concurrent), then write (serial, ordered).
        written: list = []
        emb_done = 0

        def _emb_tick() -> None:
            nonlocal emb_done
            emb_done += 1
            if progress is not None:
                progress(IngestProgress("embed", doc.title, emb_done, len(chunks)))

        if progress is not None and chunks:
            progress(IngestProgress("embed", doc.title, 0, len(chunks)))
        embeddings = map_workers(self.embedder.embed_one, [c.text for c in chunks], workers,
                                 on_complete=_emb_tick)
        for c, (vec, err) in zip(chunks, embeddings):
            if err is not None:
                if raise_errors:
                    raise err
                report.chunks_failed += 1
                report.details.append(f"embed failed [{doc.title} chunk {c.position}]: {type(err).__name__}: {err}")
                continue
            c.embedding = vec
            c.embed_model = self.embedder.model
            c.embed_dim = self.embedder.dim
            self.store.upsert_chunk(c)
            written.append(c)
            report.chunks_written += 1

        # Stage 5 -- extract (concurrent), then write (serial, ordered).
        ext_done = 0

        def _ext_tick() -> None:
            nonlocal ext_done
            ext_done += 1
            if progress is not None:
                progress(IngestProgress("extract", doc.title, ext_done, len(written)))

        if progress is not None and written:
            progress(IngestProgress("extract", doc.title, 0, len(written)))
        extractions = map_workers(
            lambda text: extract_chunk(text, self.extract_llm),
            [c.text for c in written],
            workers,
            on_complete=_ext_tick,
        )
        for c, (ex, err) in zip(written, extractions):
            if err is not None:
                if raise_errors:
                    raise err
                report.extractions_failed += 1
                report.details.append(f"extract failed [{doc.title} chunk {c.position}]: {type(err).__name__}: {err}")
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
