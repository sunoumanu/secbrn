# pgvector

pgvector is a PostgreSQL extension that adds a vector column type and nearest-neighbour
search. It is a pragmatic choice when a team already runs Postgres and only needs vector
similarity, not graph traversal.

## pgvector versus a graph database

pgvector is an alternative to [[Neo4j]] for pure [[Vector Search]]. It supports HNSW and
IVFFlat indexes. The trade-off is that pgvector has no typed relationships, so it cannot
power [[GraphRAG]]-style multi-hop reasoning on its own. For many simple "find the note
about X" workloads, that is perfectly fine.
