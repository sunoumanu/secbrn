# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) grounds a language model's output in documents
fetched at query time. Instead of relying only on parametric memory, a RAG system
retrieves relevant passages and conditions generation on them, which reduces
hallucination and lets answers cite sources.

## Pipeline

A typical RAG pipeline has two paths. The write path chunks documents and stores their
[[Embeddings]] in a vector index. The read path embeds the query, runs [[Vector Search]],
optionally adds [[Reranking]], and passes the top passages to the model. Quality depends
far more on retrieval than on the generator: if the right passage is never retrieved, no
model can answer correctly. This is why [[Hybrid Search]] and [[Retrieval Evaluation]]
matter.
