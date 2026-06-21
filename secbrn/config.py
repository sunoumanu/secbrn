"""Runtime configuration.

All knobs come from environment / a `.env` file (see `.env.example`). Centralizing
config here keeps the engine UI-agnostic: CLI, web UI (P2) and MCP server (P3) all
construct a :class:`Brain` from the same :class:`Settings`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed settings, populated from env vars prefixed ``SECBRN_``."""

    model_config = SettingsConfigDict(
        env_prefix="SECBRN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "secbrnsecbrn"
    neo4j_database: str = "neo4j"

    # Providers (model backend)
    provider: str = "ollama"  # "ollama" | "fake"
    ollama_base_url: str = "http://localhost:11434"

    # Ollama HTTP resilience. Big PDFs + slow CPUs blow past short timeouts, so these
    # are configurable and requests are retried on transient timeouts/connection drops.
    ollama_embed_timeout: float = 180.0   # seconds per embedding call
    ollama_llm_timeout: float = 600.0     # seconds per generation call
    ollama_max_retries: int = 3           # attempts per call before giving up
    ollama_retry_backoff: float = 2.0     # seconds, exponential base

    # Graph store backend, decoupled from the model provider.
    #   "auto"   -> memory iff provider == "fake", else neo4j
    #   "neo4j"  -> always the real Neo4j store
    #   "memory" -> always the in-process store (NOT persistent across processes)
    graph_backend: str = "auto"

    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    extract_model: str = "llama3.1:8b"
    answer_model: str = "llama3.1:8b"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Entity resolution thresholds (Stage 6)
    res_similarity_threshold: float = 0.86
    res_word_distance: int = 2
    res_embed_cutoff: float = 0.90
    res_llm_margin: float = 0.08

    # Retrieval
    retrieve_top_k: int = 6
    retrieve_hops: int = 2
    # Graph-aware scoring: boost a candidate chunk when its entities sit within
    # retrieve_hops of the query's entities in the graph. 0 disables it.
    graph_boost: float = 0.5
    # Lexical boost when query terms appear in a chunk's document title / section heading.
    title_boost: float = 0.4

    # Optional JSON alias-seed map for known troublemakers
    alias_seed_path: str | None = Field(default=None)


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings."""
    return Settings()
