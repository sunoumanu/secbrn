# Architecture

## 1. High-level view

```
            ┌───────────────────────── Interfaces ─────────────────────────┐
            │   CLI  (Phase 1)     Web UI (Phase 2)     MCP server (Phase 3) │
            └───────────────┬──────────────┬────────────────┬───────────────┘
                            │              │                │
                            ▼              ▼                ▼
            ┌──────────────────────── Core library (secbrn/) ───────────────┐
            │                                                                 │
            │   ingest/      extract/      resolve/      retrieve/   answer/  │
            │   loaders      schema KG     entity        hybrid      synth    │
            │                extraction    resolution    retrieval            │
            └───────────────┬─────────────────────────────────┬─────────────┘
                            │ writes                           │ reads
                            ▼                                  ▼
            ┌──────────────────────── Neo4j (single instance) ──────────────┐
            │  Property graph (typed nodes + edges)                          │
            │  + native vector index on :Chunk.embedding                     │
            │  + full-text index on :Chunk.text and :Entity.name            │
            └────────────────────────────────────────────────────────────────┘
                            ▲                                  ▲
                            │ embeddings + LLM extraction      │ embeddings + LLM answer
                            └──────────── Ollama / vLLM (local) ─────────────┘
```

Everything is local. Neo4j is the single source of truth; Ollama (or vLLM) serves both
the embedding model and the generative model.

## 2. Components

**Core library (`secbrn/`)** — the engine, UI-agnostic. Five subsystems:

- `ingest/` — source loaders + normalization to a common `Document` form, then chunking.
- `extract/` — schema-constrained KG extraction (entities, relations) per chunk.
- `resolve/` — entity resolution / dedup / canonicalization pass over the raw graph.
- `retrieve/` — hybrid retrieval (vector + full-text + graph traversal) for a query.
- `answer/` — context assembly + local LLM synthesis with citations back to sources.

**Neo4j** — stores three things in one place: the typed property graph, the vector
index over chunk embeddings, and full-text indexes. Co-locating them is what makes
hybrid (vector→graph) retrieval cheap.

**Ollama / vLLM** — local model server. Ollama is the default for simplicity; vLLM is
the option when you want higher throughput on a GPU for bulk re-ingestion.

## 3. Library decisions (ADR-style)

### ADR-1: LlamaIndex `PropertyGraphIndex` for ingestion/extraction

Chosen over LangChain `LLMGraphTransformer`. LlamaIndex's property-graph abstraction is
more mature: it ships `SchemaLLMPathExtractor` (constrain entity types, relation types,
and which connections are *allowed*), pluggable extractors, and built-in graph+vector
retrievers. The schema constraint is central to our noise-control strategy
(see [SCHEMA.md](SCHEMA.md)).

### ADR-2: `neo4j-graphrag` as the retrieval engine (optional but recommended)

Neo4j's official package bundles the retrieval patterns we'd otherwise hand-roll:
`VectorRetriever`, `VectorCypherRetriever` (vector hit → Cypher traversal),
`HybridRetriever` (vector + full-text), and `HybridCypherRetriever`. Recent versions
add a `ToolsRetriever` that lets an LLM pick among retrievers per query, and Neo4j
2026.x routes simple filters into the index `SEARCH` clause for faster filtered vector
search. Because it talks to the same Neo4j instance LlamaIndex wrote to, we can ingest
with LlamaIndex and retrieve with `neo4j-graphrag` — or vice versa.

### ADR-3: Raw `neo4j` driver for custom retrieval

For traversals the frameworks don't express well (e.g. bespoke multi-hop scoring), drop
to Cypher via the official driver. The graph is just Neo4j; nothing locks us in.

### ADR-4: Ollama default, vLLM optional

Ollama gives a one-command local setup for both embeddings and generation. vLLM is the
escape hatch for GPU throughput during large re-extraction runs.

## 4. Data model summary

- **`:Document`** — a source (note, PDF, web page, transcript) with provenance metadata.
- **`:Chunk`** — a passage of a Document; holds `text` + `embedding`; vector-indexed.
- **`:Entity`** subtypes (`:Concept`, `:Person`, `:Tool`, `:Org`, ...) — canonical nodes.
- **Edges** between entities are typed (`RELATES_TO`, `PART_OF`, `USES`, ...) and
  constrained by the schema. Chunks link to the entities they mention via `:MENTIONS`.

Full definition, constraints, and indexes in [SCHEMA.md](SCHEMA.md).

## 5. Configuration

All runtime config via `.env` / `secbrn/config.py`:

- Neo4j URI / user / password / database
- Embedding model name + dimension
- Extraction LLM and answer LLM (can differ — a stronger model for extraction)
- Chunk size / overlap
- Allowed schema (node labels, relation types) — see [SCHEMA.md](SCHEMA.md)
- Resolution thresholds (similarity, word distance)
