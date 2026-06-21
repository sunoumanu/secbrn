# Vector Search

Vector search finds passages whose embedding is closest to the query embedding under a
similarity function, usually cosine. It captures semantic similarity, so a query and a
passage can match even when they share no words.

## Approximate nearest neighbours

Exact search is O(n) per query, so production systems use an approximate index such as
[[HNSW]] or IVF. FAISS and Neo4j both provide vector indexes. Vector search excels at
paraphrase and synonym matching but struggles with rare exact tokens like identifiers or
product names — that weakness is exactly what [[BM25 and Lexical Search]] covers, which
is why the two are fused in [[Hybrid Search]].
