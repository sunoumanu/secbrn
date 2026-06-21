# Full Pipeline: Ingestion → Retrieval

This is the heart of SecBrn. Eight stages, grouped into **write-path** (1–6, build the
graph) and **read-path** (7–8, answer a question). Each stage lists *input → output*,
the failure mode it guards against, and the concrete library call.

```
WRITE PATH                                              READ PATH
1 Load ──► 2 Normalize ──► 3 Chunk ──► 4 Embed          7 Retrieve ──► 8 Synthesize
                                  └──► 5 Extract KG          (vector + FT + graph)
                                          └──► 6 Resolve
```

---

## STAGE 1 — Load

**Input:** a file path, folder, or URL. **Output:** raw bytes/text + source metadata.

Source-specific loaders, all producing the same internal shape:

- **Notes / Markdown** — read file; keep frontmatter, headings, and `[[wikilinks]]`
  (wikilinks become candidate explicit edges — a free, high-quality signal).
- **PDFs / documents** — extract text + page numbers (`pymupdf`/`unstructured`); OCR
  scanned PDFs as a fallback. Keep page spans for citations.
- **Web / bookmarks** — fetch + readability extraction to clean article text; record
  canonical URL, title, fetch date.
- **Chats / transcripts** — parse role-tagged turns (LLM exports, meeting transcripts);
  preserve speaker + timestamp so turns stay attributable.

Each loader emits `{source_id, source_type, uri, title, created_at, raw_text, spans}`.

**Guards against:** losing provenance. Every downstream node must trace back to a source
location for citations and re-ingestion.

---

## STAGE 2 — Normalize & deduplicate sources

**Input:** raw loader output. **Output:** a clean `Document` + a content hash.

- Strip boilerplate, normalize whitespace/encoding, drop nav chrome from web pages.
- Compute a **content hash**. If the hash already exists, skip (idempotent ingestion —
  re-running `ingest` on a folder is safe and cheap).
- Detect *updates*: same `uri`, new hash → version the Document and mark stale chunks
  for re-extraction rather than duplicating.

**Guards against:** duplicate documents silently inflating the graph; re-ingestion
churn.

---

## STAGE 3 — Chunk

**Input:** `Document`. **Output:** ordered `Chunk`s with stable IDs and back-pointers.

- Structure-aware splitting: prefer Markdown headings / PDF page boundaries / transcript
  turns over blind character windows. Fall back to sentence-window splitting
  (~512–1024 tokens, ~10–15% overlap).
- Each chunk records `document_id`, `position`, and the original `span` (page/line/turn)
  for citation.

**Guards against:** mid-sentence cuts that wreck both embedding quality and extraction.
Chunk size is a tuned knob — too big dilutes embeddings, too small fragments relations.

---

## STAGE 4 — Embed

**Input:** chunk text. **Output:** `:Chunk` nodes with `embedding` written to Neo4j's
native vector index.

- Embed with the local model (`nomic-embed-text` via Ollama, 768-dim, or a
  `sentence-transformers` model). Batch for throughput.
- Write `(:Document)-[:HAS_CHUNK]->(:Chunk {text, embedding, span})`.
- Create/ensure the vector index on `:Chunk(embedding)` and full-text index on
  `:Chunk(text)`.

**Guards against:** dimension/model drift — store the model name + dim on each chunk so a
later model change is detectable and triggers re-embed, not silent mismatch.

---

## STAGE 5 — Extract knowledge graph (schema-constrained)

**Input:** chunk text. **Output:** typed entities + typed relations, attached to chunks.

This is where noise enters, so it is **constrained, not free-form**:

- Use LlamaIndex **`SchemaLLMPathExtractor`** with an explicit schema
  (allowed node labels, allowed relation types, and the *valid (subject, relation,
  object) triples*). The LLM may only emit graph data conforming to the schema. This
  trades some recall for a far more consistent graph — the right trade for a long-lived
  brain. (See [SCHEMA.md](SCHEMA.md) for the schema itself.)
- Run extraction with a capable local model (extraction can use a stronger/slower model
  than answering, since it runs offline at ingest time).
- Every extracted entity/relation keeps a **provenance edge to its source chunk**
  (`(:Chunk)-[:MENTIONS]->(:Entity)`), so any later answer can cite where a fact came
  from and so bad extractions can be traced and pruned.
- Promote **explicit signals** from Stage 1 (Markdown `[[wikilinks]]`, hyperlinks
  between docs) into high-confidence edges — these are cleaner than LLM guesses.

**Guards against:** the core failure mode — spurious relationships and uncontrolled node
labels. Schema constraint is the first line of defense.

---

## STAGE 6 — Entity resolution / dedup / canonicalization

**Input:** the raw, noisy extracted graph. **Output:** a clean graph with merged
duplicates and canonical names.

LLM extraction *will* produce `pgvector` / `Pgvector` / `PGVector` as three nodes. This
dedicated pass is non-optional:

1. **Blocking** — group candidate duplicates cheaply (same label + similar normalized
   name, shared aliases, or embedding-near names) to avoid O(n²) comparison.
2. **Similarity scoring** — within a block, score pairs by string distance
   (normalized name + `word_distance`) **and** embedding similarity of the entity and
   its context. Tune `similarity_threshold` / `word_distance` to catch duplicates
   without over-merging.
3. **Merge** — collapse confirmed duplicates into one canonical node; keep variants as
   `aliases`; rewire all edges to the canonical node; preserve provenance.
4. **Canonicalize** — pick a canonical surface form (prefer the most frequent or an
   explicit alias map for known tools/people).
5. **Optional LLM adjudication** — for ambiguous pairs above a margin, ask the local LLM
   "are these the same entity?" with context — used sparingly, it's expensive.

Entity disambiguation has no perfect solution; expect false positives, so make merges
**reversible** (record merge history) and run this incrementally as new data arrives.

**Guards against:** the second core failure mode — fragmented duplicate entities that
scatter a concept's relationships across several nodes and ruin multi-hop retrieval.

> Stages 5–6 are also where most tuning time goes. Budget for it. A clean small graph
> beats a large messy one.

---

## STAGE 7 — Retrieve (hybrid: vector + full-text + graph)

**Input:** a user question. **Output:** a ranked, deduplicated context bundle.

Retrieval composes three signals (via `neo4j-graphrag` retrievers or custom Cypher):

1. **Vector search** — embed the query, find nearest `:Chunk`s (Neo4j vector index;
   2026.x routes simple filters into the in-index `SEARCH` clause for speed).
2. **Full-text / keyword** — full-text index on chunk text and entity names catches
   exact terms, rare tokens, and identifiers that embeddings blur (`HybridRetriever`
   fuses vector + full-text).
3. **Graph expansion** — from the entities mentioned in the top chunks, traverse typed
   relationships 1–2 hops (`VectorCypherRetriever` / custom Cypher). This is the step
   that answers "how does X relate to Y" — it pulls in connected facts a pure vector
   store can't reach.

Then **fuse and rerank**: combine the chunk hits and graph-expanded context, dedup,
score by relevance + graph proximity, and (optionally) rerank with a cross-encoder or
the local LLM. For complex queries, a `ToolsRetriever` can let the LLM choose which
retriever(s) to fire per question.

**Output bundle:** the selected chunks (with spans), the relevant subgraph
(entities + typed edges), and provenance for every item.

---

## STAGE 8 — Synthesize answer

**Input:** the question + context bundle. **Output:** an answer with citations.

- Assemble a prompt that gives the local LLM **both** the retrieved chunks *and* a
  serialized view of the relevant subgraph (entities and how they connect) — the graph
  structure is what lets the model reason across documents rather than summarize one.
- Generate with the local answer model (Ollama/vLLM).
- **Cite** every claim back to its source chunk/document span. Because provenance was
  preserved from Stage 1, citations are exact (file + page/line/turn).
- Optionally return the **subgraph used** so the UI can show *why* the answer connects
  the facts it did.

**Guards against:** ungrounded generation. No citation → flag it. The brain should be
auditable.

---

## End-to-end example

> **Q:** "How does my note on retrieval relate to what I read about rerankers?"

1. Embed query → vector hits in the retrieval note and the reranker article.
2. Full-text catches the literal term "reranker".
3. Both chunks mention entity `Reranking`; graph expansion finds
   `Reranking -[:IMPROVES]-> Retrieval` and `Reranking -[:USED_IN]-> RAG`, plus a
   `Cross-Encoder` node connected to both sources.
4. Fuse: chunks + that subgraph.
5. LLM answers, citing the note (line span) and the article (URL + section), and
   explains the connection via the `Reranking → Retrieval` path — something neither
   document stated outright but the *graph* did.
