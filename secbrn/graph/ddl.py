"""Neo4j DDL — constraints + vector + full-text indexes (docs/SCHEMA.md §4).

The ``entity.name IS UNIQUE`` constraint is what makes the Stage-6 resolution merge
enforceable: two nodes cannot both claim the canonical name.
"""

from __future__ import annotations

CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_nm IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
]

FULLTEXT_INDEXES: list[str] = [
    "CREATE FULLTEXT INDEX chunk_ft IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX entity_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.aliases]",
]


def vector_index_ddl(dim: int) -> str:
    """Vector index; dimension must match the embedding model (768 for nomic-embed-text)."""
    return (
        "CREATE VECTOR INDEX chunk_vec IF NOT EXISTS "
        "FOR (c:Chunk) ON (c.embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {dim}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


def all_ddl(dim: int) -> list[str]:
    return [*CONSTRAINTS, vector_index_ddl(dim), *FULLTEXT_INDEXES]
