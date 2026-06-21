# Retrieval Evaluation

Retrieval evaluation measures whether the right passages come back for a query, using a
gold set of queries mapped to relevant documents. Precision@k is misleading when the
number of relevant documents is much smaller than k.

## Better metrics

precision@R, MAP, and nDCG@k are rank-aware and not distorted by that artifact. A gold set
should grow from real failures. Evaluation is how you safely tune chunk size,
[[Reranking]], and [[Hybrid Search]] — without it, changes are guesses. It applies equally
to [[GraphRAG]] retrieval.
