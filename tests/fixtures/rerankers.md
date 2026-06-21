# Rerankers

A reranker takes the candidate passages from first-stage Retrieval and scores each
one against the query. Rerankers improve Retrieval quality substantially.

The most common reranker is a Cross-Encoder, which jointly encodes the query and a
passage. Cross-Encoder models are slower than bi-encoders but far more accurate.

Reranking is used in RAG pipelines after vector search.
