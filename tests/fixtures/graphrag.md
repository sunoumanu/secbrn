# GraphRAG

GraphRAG combines Retrieval with a knowledge graph so the model can do multi-hop
reasoning across documents instead of summarising a single chunk.

## Why a graph

Graph expansion traverses typed relationships in Neo4j to pull in connected facts,
which is what lets GraphRAG answer "how does X relate to Y".
