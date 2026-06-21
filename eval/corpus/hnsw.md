# HNSW

Hierarchical Navigable Small World (HNSW) is a graph-based index for approximate nearest-
neighbour search. It builds a layered proximity graph and greedily walks it, giving
sub-linear query time with high recall.

## Tuning

Two parameters dominate: M controls graph connectivity and efSearch controls how widely
the search explores at query time. Higher efSearch raises recall at the cost of latency.
HNSW is used inside [[pgvector]], FAISS, and [[Neo4j]] to make [[Vector Search]] fast at
scale.
