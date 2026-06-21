# SecBrn — Local Graph Second Brain for LLMs

A single-user, fully local "second brain": ingest your notes, PDFs, web clips, and
chat transcripts into a **Neo4j knowledge graph + vector index**, then query it with a
**local LLM** that reasons across the typed relationships between your facts — not just
retrieves the nearest chunk.

Everything runs on your machine. No data leaves the box.

---

## Why a graph (the honest tradeoff)

A graph shines when **the connections between facts are the point** — multi-hop
questions, "how does everything relate," reasoning across documents accumulated over
years. If 90% of your queries are "find the note about X," a plain vector store is
simpler and a graph is just operational overhead.

SecBrn is built for the first case. The typed relationships are exactly the structure
that lets the LLM *reason* instead of just *retrieve*. The cost is real and we pay it
deliberately: LLM-extracted graphs are noisy (e.g. `pgvector` / `Pgvector` / `PGVector`
become three nodes; spurious edges; missed links). So this project treats
**schema constraint + entity resolution as first-class pipeline stages**, not
afterthoughts. A messy graph retrieves *worse* than no graph.

---

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Graph store | **Neo4j Community 5.x** | Free, single-user friendly, native vector index |
| Graph framework | **LlamaIndex `PropertyGraphIndex`** | Most mature property-graph abstraction; schema-constrained extraction |
| Retrieval (alt) | **`neo4j-graphrag`** (Neo4j official) | Bundles vector / vector-cypher / hybrid retrievers so you don't hand-roll them |
| Embeddings | **Ollama `nomic-embed-text`** (or `sentence-transformers`) | Local, 768-dim |
| LLM (extract + answer) | **Ollama** (e.g. `llama3.1`, `qwen2.5`) or **vLLM** | Local generation |
| Direct queries | **`neo4j` Python driver** | For custom Cypher retrieval |

We standardize on **LlamaIndex for ingestion/extraction** and keep **`neo4j-graphrag`
as the retrieval engine option** — both speak to the same Neo4j instance, so they
compose. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the decision record.

---

## Documents

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — components, data flow, library decisions
- **[docs/PIPELINE.md](docs/PIPELINE.md)** — full ingestion → retrieval pipeline, stage by stage
- **[docs/SCHEMA.md](docs/SCHEMA.md)** — graph schema, constraints, entity resolution & dedup
- **[docs/FEATURES.md](docs/FEATURES.md)** — complete feature catalog (MVP → advanced)
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — phased build plan (CLI → Web UI → MCP server)

---

## Quickstart (target end-state of Phase 1)

```bash
# 1. Infra
docker compose up -d            # Neo4j + (optional) Ollama
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 2. Install
pip install -e .

# 3. Configure
cp .env.example .env            # Neo4j creds, model names, paths

# 4. Ingest
secbrn ingest ./my-notes/       # folder of md/pdf/html/json
secbrn ingest --url https://...

# 5. Ask
secbrn ask "How does my note on retrieval relate to what I read about rerankers?"
```

## Status

**Phase 0 + Phase 1 implemented.** The full 8-stage pipeline (load → normalize →
chunk → embed → extract → resolve → retrieve → synthesize), the `secbrn` CLI, and an
importable `Brain` Python API are in place, with a per-stage test suite and an
end-to-end multi-hop verification test.

The core engine runs on the official `neo4j` driver behind a `GraphStore` interface,
so `neo4j-graphrag` / LlamaIndex retrievers can be plugged in later (ARCHITECTURE
ADR-2/3). Models are pluggable (`secbrn/providers/`), defaulting to Ollama.

### Offline mode (no Neo4j / no Ollama)

Set `SECBRN_PROVIDER=fake` to run the whole pipeline against a deterministic
in-memory store + fake models — used by the test suite. Note the in-memory store is
**not persistent across processes**; use the Neo4j backend (default) for a real brain.

```bash
pip install -e ".[dev]"
python -m pytest                 # full offline test suite
python -m secbrn.healthcheck     # checks Neo4j + models + indexes
secbrn ingest ./my-notes/        # real run (needs Neo4j + Ollama up)
secbrn ask "How does my retrieval note relate to rerankers?"
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for what's next (Phase 2 web UI).

## Measuring quality (eval harness)

`secbrn eval` scores the brain against a gold set so you can defend any
schema/threshold/chunking change with a number instead of a vibe.

```bash
secbrn eval --gold eval/gold.json            # ingests the gold corpus, then scores
secbrn eval --gold eval/gold.json --show-cases   # per-query retrieval detail
secbrn eval --use-existing                    # score the data already in the store
```

Three layers are measured (evaluate any subset you have labels for):

- **Retrieval** — precision@k, **precision@R**, recall@k, **MAP**, **nDCG@k**, MRR, hit@k
  over `query → relevant-doc` labels. Read precision@R / nDCG@k first; plain precision@k
  is capped at ~`relevant/k` and looks artificially low on small corpora.
- **Extraction** — micro precision/recall/F1 of entities and triples vs. hand-labeled text.
- **Resolution** — pairwise precision/recall/F1 of should-merge / should-not-merge pairs
  (FP = over-merges, the dangerous failure; FN = missed duplicates).

Retrieval is scored against an **isolated** in-memory copy of just the gold corpus, so
other data in your Neo4j can't dilute precision (`--use-existing` scores the live store).
Graph-aware scoring (`SECBRN_GRAPH_BOOST`) boosts chunks whose entities sit near the
query's entities in the graph.

The gold set is a small JSON file (`eval/gold.json`). Full guidance — metric meanings,
tuning knobs, how to build a real gold set — is in **[docs/EVAL.md](docs/EVAL.md)**.
