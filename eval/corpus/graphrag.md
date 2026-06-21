# GraphRAG

GraphRAG augments [[Retrieval-Augmented Generation]] with a [[Knowledge Graphs|knowledge
graph]] so the model can do multi-hop reasoning across documents instead of summarising a
single chunk. After vector hits land, the system expands along typed relationships to pull
in connected facts.

## Graph expansion

Graph expansion traverses one or two hops from the entities mentioned in the top
passages, surfacing facts that no single passage states. This is what answers "how does X
relate to Y" questions. GraphRAG relies on [[Neo4j]] for storage and on clean
[[Entity Resolution]] so a concept is not split across duplicate nodes.
