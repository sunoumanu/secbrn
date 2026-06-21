# Entity Resolution

Entity resolution merges duplicate entities that refer to the same thing — for example
pgvector, Pgvector, and PGVector — into one canonical node, keeping the variants as
aliases. Without it, a concept's relationships scatter across several nodes and multi-hop
retrieval breaks.

## How it works

Resolution blocks candidates cheaply, scores pairs by string distance and embedding
similarity, then merges above a threshold. Over-merging is the dangerous failure, so
merges are kept reversible. Resolution quality directly determines whether
[[Knowledge Graphs]] and [[GraphRAG]] stay clean as they grow.
