# Graph Schema & Entity Resolution

The schema is the project's main lever against extraction noise. Constrain *what the LLM
is allowed to create*, and you trade a little recall for a graph that stays clean enough
to reason over for years.

## 1. Node labels

### Structural (not LLM-extracted — created deterministically)

| Label | Key properties | Purpose |
|---|---|---|
| `:Document` | `id, source_type, uri, title, created_at, content_hash, version` | A source |
| `:Chunk` | `id, text, embedding, span, position, embed_model, embed_dim` | A passage; vector-indexed |

### Semantic (LLM-extracted, constrained to this closed set)

| Label | Examples | Notes |
|---|---|---|
| `:Concept` | "retrieval", "entity resolution" | Default for abstract ideas |
| `:Tool` | "Neo4j", "Ollama", "pgvector" | Software/libraries/products |
| `:Person` | "Andrej Karpathy" | People |
| `:Org` | "Anthropic", "Neo4j Inc" | Organizations |
| `:Topic` | "GraphRAG", "RAG" | Broad subject areas |
| `:Event` | "release of Neo4j 2026.01" | Time-anchored happenings |
| `:Place` | "San Francisco" | Locations |

> Keep this set **small and closed**. Every label you add multiplies the ways extraction
> can disagree with itself. Add labels only when a real query needs the distinction.

All semantic nodes also carry a shared `:Entity` label + common props:
`name` (canonical), `aliases: [..]`, `summary`, `created_at`, `mention_count`.

## 2. Relationship types (closed set)

| Type | Typical (subject → object) | Meaning |
|---|---|---|
| `RELATES_TO` | Entity → Entity | Generic association (fallback) |
| `PART_OF` | Concept → Topic | Composition / hierarchy |
| `USES` | Tool/Person → Tool | Usage / dependency |
| `IMPROVES` | Concept → Concept | One enhances another |
| `ALTERNATIVE_TO` | Tool → Tool | Competing/substitute |
| `AUTHORED_BY` | Document → Person | Authorship |
| `MENTIONS` | Chunk → Entity | **Provenance** (every fact traces to a chunk) |
| `HAS_CHUNK` | Document → Chunk | Structural |
| `LINKS_TO` | Document → Document | Explicit `[[wikilink]]` / hyperlink |
| `SAME_AS` (history) | Entity → Entity | Records a reversible merge |

## 3. Schema validation (the allowed-triples table)

`SchemaLLMPathExtractor` is given not just labels and relations but the **valid triples**
— which (subject_label, relation, object_label) combinations are permitted. Anything off
this list is discarded at extraction time.

```python
# illustrative — the actual list lives in secbrn/extract/schema.py
validation_schema = [
    ("Tool",    "ALTERNATIVE_TO", "Tool"),
    ("Tool",    "USES",           "Tool"),
    ("Concept", "IMPROVES",       "Concept"),
    ("Concept", "PART_OF",        "Topic"),
    ("Person",  "USES",           "Tool"),
    ("Document","AUTHORED_BY",    "Person"),
    ("Entity",  "RELATES_TO",     "Entity"),   # permissive fallback
    # ...
]
```

`strict=True` so the extractor drops non-conforming triples instead of coercing them.

## 4. Neo4j constraints & indexes (DDL)

```cypher
// Uniqueness / existence
CREATE CONSTRAINT doc_id    IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT chunk_id  IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_nm IF NOT EXISTS FOR (e:Entity)   REQUIRE e.name IS UNIQUE;

// Vector index (dim must match embed model; 768 for nomic-embed-text)
CREATE VECTOR INDEX chunk_vec IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

// Full-text indexes (hybrid retrieval)
CREATE FULLTEXT INDEX chunk_ft  IF NOT EXISTS FOR (c:Chunk)  ON EACH [c.text];
CREATE FULLTEXT INDEX entity_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases];
```

The `entity.name IS UNIQUE` constraint is what makes the resolution merge (Stage 6)
enforceable — two nodes can't both claim the canonical name.

## 5. Entity resolution policy

Detailed flow is in [PIPELINE.md](PIPELINE.md) Stage 6. Schema-level rules:

- **Canonical name** is the unique key; every surface variant goes into `aliases`.
- **Merge is reversible** — record a `:SAME_AS` history edge + a merge log so a bad merge
  can be undone.
- **Thresholds are config**, not hardcoded: `similarity_threshold`, `word_distance`,
  embedding-similarity cutoff, and an `llm_adjudication_margin` band where the local LLM
  breaks ties.
- **Alias seed list** for known troublemakers (`pgvector` family, common tool casings)
  short-circuits the fuzzy logic for high-value entities.
- **Run incrementally** — resolve newly-added entities against existing canonical nodes
  on each ingest, plus a periodic full sweep.

## 6. Evolving the schema

Schema changes are migrations:

1. Edit `schema.py` (labels / relations / valid triples).
2. Decide: re-extract affected documents, or map old → new labels in place.
3. Bump a `schema_version` property on `:Document` so you know what was extracted under
   which schema.

Resist the urge to make the schema large early. Start with the closed sets above, watch
which real questions fail for lack of a type, and grow deliberately.
