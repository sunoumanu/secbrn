# Retrieval Notes

Retrieval is the core of any RAG system. Good retrieval finds the most relevant
passages for a query before the model answers.

## Reranking

Reranking improves Retrieval by reordering candidate passages with a stronger model.
A Cross-Encoder is the usual tool for reranking. See [[Rerankers]] for details.

## Graph expansion

Graph expansion uses Neo4j to traverse typed relationships. It complements vector
search by pulling in connected facts.
