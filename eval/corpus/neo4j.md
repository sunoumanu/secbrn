# Neo4j

Neo4j is a graph database that stores typed nodes and relationships and, since version 5,
ships a native vector index. Co-locating the property graph, the vector index, and
full-text indexes in one store is what makes hybrid vector-to-graph retrieval cheap.

## Role in the stack

Neo4j is the single source of truth for [[Knowledge Graphs]] and [[GraphRAG]]. It is an
alternative to [[pgvector]] when the connections between facts matter, not just nearest-
neighbour lookup. Queries are written in Cypher, and the vector index dimension must match
the [[Embedding Models|embedding model]].
