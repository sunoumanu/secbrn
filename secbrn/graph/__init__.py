"""Graph store factory."""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.graph.base import GraphStore
from secbrn.graph.memory import InMemoryStore

__all__ = ["GraphStore", "InMemoryStore", "get_store", "resolve_backend"]


def resolve_backend(settings: Settings) -> str:
    """Resolve the effective store backend name ("neo4j" | "memory")."""
    backend = (settings.graph_backend or "auto").lower()
    if backend == "auto":
        return "memory" if settings.provider == "fake" else "neo4j"
    if backend not in ("neo4j", "memory"):
        raise ValueError(
            f"SECBRN_GRAPH_BACKEND must be auto|neo4j|memory, got {backend!r}"
        )
    return backend


def get_store(settings: Settings) -> GraphStore:
    """Return the configured store.

    The store backend is chosen by ``SECBRN_GRAPH_BACKEND`` (default ``auto``), NOT by
    the model provider. ``memory`` is the in-process store and is *not* persistent
    across processes — if you want data to show up in Neo4j you must be on ``neo4j``.
    """
    if resolve_backend(settings) == "memory":
        return InMemoryStore()
    from secbrn.graph.neo4j_store import Neo4jStore

    return Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        embed_dim=settings.embed_dim,
    )
