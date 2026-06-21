# Reranking

Reranking reorders first-stage candidates with a stronger, slower model that scores each
query–passage pair jointly. It improves precision at the top of the list, which is what
the generator in [[Retrieval-Augmented Generation]] actually reads.

## Cross-encoders

A cross-encoder encodes the query and passage together, unlike the bi-encoder used for
[[Vector Search]], making it far more accurate but too expensive to run over the whole
corpus. So rerankers run only on the fused top-k from [[Hybrid Search]]. Reranking
improves retrieval and is measured with [[Retrieval Evaluation]].
