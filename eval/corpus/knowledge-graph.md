# Knowledge Graphs

A knowledge graph stores facts as typed nodes and typed relationships, so the connections
between facts are first-class. This lets a system reason over multi-hop paths rather than
only matching a single passage.

## Why typed edges

Typed relationships such as USES, IMPROVES, and ALTERNATIVE_TO let queries traverse from
one entity to related ones. The cost is that LLM-extracted graphs are noisy, so a clean
schema and [[Entity Resolution]] are essential. Knowledge graphs underpin [[GraphRAG]]
and are commonly stored in [[Neo4j]].
