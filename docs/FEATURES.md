# Feature Catalog

Grouped by subsystem. Each feature tagged with target phase
(**[P1]** CLI core · **[P2]** Web UI · **[P3]** MCP/advanced). See
[ROADMAP.md](ROADMAP.md) for sequencing.

## Ingestion

- **[P1] Multi-source loaders** — Markdown/notes, PDFs & documents, web pages/bookmarks,
  chat & meeting transcripts; all normalized to a common `Document` shape.
- **[P1] Provenance capture** — every Document/Chunk keeps source URI, title, date, and
  exact span (page / line / turn) for citation and re-ingestion.
- **[P1] Idempotent ingestion** — content-hash dedup; re-running `ingest` on a folder is
  safe and skips unchanged files.
- **[P1] Update detection & versioning** — changed source → new version, stale chunks
  re-extracted instead of duplicated.
- **[P1] Structure-aware chunking** — split on headings / pages / turns with overlap;
  configurable size.
- **[P2] Watch folders** — auto-ingest on file change.
- **[P2] Web clipper** — quick "save this URL" endpoint/bookmarklet.
- **[P3] Scheduled re-ingest** — periodic refresh of watched sources.

## Extraction & graph construction

- **[P1] Schema-constrained extraction** — `SchemaLLMPathExtractor` with closed label /
  relation / valid-triple sets ([SCHEMA.md](SCHEMA.md)).
- **[P1] Explicit-link promotion** — Markdown `[[wikilinks]]` and hyperlinks become
  high-confidence edges.
- **[P1] Provenance edges** — `(:Chunk)-[:MENTIONS]->(:Entity)` on every extracted fact.
- **[P2] Extraction model choice** — use a stronger/slower local model at ingest time.
- **[P3] Incremental & dynamic schema** — propose new types from data for human approval.

## Entity resolution

- **[P1] Dedup pass** — blocking + string + embedding similarity to merge duplicates.
- **[P1] Canonicalization & aliases** — single canonical node, variants kept as aliases.
- **[P1] Alias seed list** — hard-coded canonical forms for known troublemakers.
- **[P2] Reversible merges** — merge history + undo.
- **[P2] LLM tie-break adjudication** — local LLM resolves ambiguous pairs.
- **[P3] Continuous resolution** — incremental on ingest + periodic full sweep.

## Retrieval

- **[P1] Vector retrieval** — Neo4j native vector index over chunk embeddings.
- **[P1] Hybrid retrieval** — vector + full-text fusion.
- **[P1] Graph-expansion retrieval** — traverse typed relations 1–2 hops from hit
  entities (vector-cypher).
- **[P2] Rerank** — cross-encoder or LLM rerank of fused candidates.
- **[P2] Filtered retrieval** — by source type, date, document, tag (in-index `SEARCH`).
- **[P3] Tool-routed retrieval** — `ToolsRetriever`; LLM picks retrievers per query.
- **[P3] Multi-hop / agentic retrieval** — iterative retrieve-reason-retrieve loops.

## Answering

- **[P1] Grounded synthesis** — local LLM answers from chunks **+** serialized subgraph.
- **[P1] Inline citations** — every claim cites a source span; uncited claims flagged.
- **[P2] "Show the subgraph"** — return the entities/edges used so the user sees *why*.
- **[P2] Conversational memory** — follow-up questions keep context.
- **[P3] Saved/streaming answers** — persistent threads, token streaming.

## Interfaces

- **[P1] CLI** — `ingest`, `ask`, `search`, `stats`, `resolve`, `reindex`.
- **[P1] Python API** — import `secbrn` as a library.
- **[P2] Local web UI** — chat pane + results with citations + interactive graph
  explorer (visualize neighborhoods, click to expand).
- **[P3] MCP server** — expose `search` / `ask` / `add_note` as MCP tools so Claude and
  other LLMs can query the brain directly.

## Operations & quality

- **[P1] Config-driven** — `.env` for models, thresholds, chunking, schema path.
- **[P1] Local-only guarantee** — no network egress for data; all models local.
- **[P1] Graph stats** — node/edge counts, duplicate estimates, orphan chunks.
- **[P2] Backup/restore** — Neo4j dump + export of config & schema.
- **[P2] Eval harness** — gold Q→A/context set to measure retrieval quality across
  schema/threshold changes.
- **[P3] Health dashboard** — extraction noise metrics, dedup rate, answer-citation rate.
- **[P3] Pluggable backends** — swap embedding/LLM models; optional vLLM serving.
