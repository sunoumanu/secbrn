# Development Roadmap

Phased so each phase ships something usable. The core library is built once in Phase 1;
Phase 2 and 3 are interfaces over the same engine.

```
Phase 0  Foundations      → infra + skeleton
Phase 1  CLI + core       → end-to-end brain you can ingest & query
Phase 2  Web UI           → chat + graph explorer + quality tooling
Phase 3  MCP + advanced   → expose to LLMs, agentic retrieval, scale
```

Effort estimates assume one developer; treat as relative, not calendar-exact.

---

## Phase 0 — Foundations (~2–3 days)

**Goal:** infra runs, repo skeleton, models pulled.

- [ ] `docker-compose.yml`: Neo4j Community 5.x (volumes, vector-index enabled) + Ollama.
- [ ] Pull models: `nomic-embed-text`, an extraction LLM, an answer LLM.
- [ ] Repo layout: `secbrn/{ingest,extract,resolve,retrieve,answer}`, `tests/`, `docs/`.
- [ ] `config.py` + `.env.example`; Neo4j connection smoke test.
- [ ] Apply DDL from [SCHEMA.md](SCHEMA.md) (constraints, vector + full-text indexes).

**Exit:** `python -m secbrn.healthcheck` confirms Neo4j + both models reachable and
indexes present.

---

## Phase 1 — CLI + core engine (~2–3 weeks)

**Goal:** a real, local second brain usable from the terminal. This is the milestone
that delivers the [PIPELINE.md](PIPELINE.md) end to end.

### 1a. Write path
- [ ] Loaders: Markdown, PDF, web, transcript → common `Document` (Stage 1).
- [ ] Normalize + content-hash dedup + versioning (Stage 2).
- [ ] Structure-aware chunking (Stage 3).
- [ ] Embedding + Neo4j write + vector/full-text index population (Stage 4).
- [ ] `SchemaLLMPathExtractor` extraction with the closed schema + provenance edges +
      wikilink promotion (Stage 5).
- [ ] Entity resolution pass: blocking, similarity, merge, canonicalize, alias seed
      list (Stage 6).

### 1b. Read path
- [ ] Vector + hybrid + graph-expansion retrieval via `neo4j-graphrag` (Stage 7).
- [ ] Grounded answer synthesis with inline citations (Stage 8).

### 1c. CLI + library
- [ ] `secbrn ingest <path|--url>`, `ask`, `search`, `stats`, `resolve`, `reindex`.
- [ ] Importable Python API mirroring the CLI.

### 1d. Quality
- [ ] Unit tests per stage; small fixture corpus (a few md/pdf/html/transcript files).
- [ ] **Verification:** end-to-end test — ingest fixtures, assert the known multi-hop
      question returns the expected linked facts with correct citations.

**Exit:** ingest a real folder of your notes + a few PDFs + URLs, ask a multi-hop
question, get a cited answer that uses graph connections. Tune chunk size + resolution
thresholds against results.

---

## Phase 2 — Local web UI + quality tooling (~2–3 weeks)

**Goal:** pleasant daily-use UX and the tooling to keep the graph clean as it grows.

- [ ] FastAPI backend wrapping the core library (`/ingest`, `/ask`, `/search`, `/graph`).
- [ ] Frontend: chat pane with streamed, cited answers.
- [ ] **Graph explorer** — visualize an entity's neighborhood, click to expand hops,
      jump from a citation to its source.
- [ ] "Show the subgraph used" toggle on answers.
- [ ] Watch-folder + web-clipper ingestion.
- [ ] Conversational memory for follow-ups.
- [ ] Reranking + filtered retrieval (source type / date / document).
- [ ] Reversible merges UI + LLM tie-break adjudication for resolution.
- [ ] **Eval harness** — gold question set; compare retrieval/answer quality across
      schema and threshold changes (this is how you safely tune).
- [ ] Backup/restore.

**Exit:** you use the web UI as your daily brain; eval harness gives a number to defend
schema/threshold changes against.

---

## Phase 3 — MCP server + advanced retrieval (~2–4 weeks)

**Goal:** let other LLMs use the brain, and push retrieval quality + scale.

- [ ] **MCP server** exposing `search`, `ask`, `add_note` as tools — Claude and other
      LLMs query your brain directly.
- [ ] Tool-routed retrieval (`ToolsRetriever`): LLM picks retrievers per query.
- [ ] Multi-hop / agentic retrieval (iterative retrieve→reason→retrieve).
- [ ] Continuous + scheduled resolution and re-ingest.
- [ ] Dynamic schema proposals (data suggests new types → human approves).
- [ ] Health dashboard: extraction-noise metrics, dedup rate, citation rate.
- [ ] Optional vLLM serving for GPU throughput; pluggable model backends.

**Exit:** the brain is an MCP tool in your LLM workflows; retrieval handles
"how does everything relate"–class questions across a large, clean graph.

---

## Cross-cutting principles

1. **Schema + resolution are first-class.** Budget real time for Stages 5–6 tuning; a
   clean small graph beats a large messy one.
2. **Provenance everywhere.** No node without a source span; no answer without citations.
3. **Local-only.** No data egress; all models local.
4. **Idempotent & reversible.** Re-ingest safely; undo bad merges.
5. **Measure before tuning.** Once the eval harness exists (P2), changes are judged by
   it, not vibes.

## Key risks & mitigations

| Risk | Mitigation |
|---|---|
| Noisy LLM extraction (the big one) | Schema constraint + dedicated resolution + provenance pruning |
| Over-merging entities | Reversible merges, conservative thresholds, alias seeds, LLM tie-break |
| Graph overhead not worth it for simple lookups | Hybrid retrieval still gives plain vector search; graph only adds when connections matter |
| Local model quality for extraction | Use stronger model at ingest time; extraction is offline so latency is OK |
| Schema too rigid / too loose | Start small + closed, grow via migrations driven by failing real queries |
