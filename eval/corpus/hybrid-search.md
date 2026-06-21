# Hybrid Search

Hybrid search combines dense [[Vector Search]] with sparse [[BM25 and Lexical Search]] so
a system benefits from both semantic recall and exact-term precision. The two ranked
lists are merged, commonly with reciprocal rank fusion (RRF), which needs no score
calibration.

## When it helps

Hybrid retrieval helps most on queries that mix a concept with a specific identifier —
for example "how does HNSW tune efSearch". Dense retrieval finds the HNSW passage by
meaning; lexical retrieval guarantees the rare token efSearch is matched. After fusion,
[[Reranking]] can reorder the candidates, and [[Retrieval Evaluation]] tells you whether
the fusion actually helped.
