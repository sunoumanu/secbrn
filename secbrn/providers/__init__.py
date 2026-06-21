"""Pluggable model providers (embeddings + generation).

Default is Ollama (docs ADR-4). A deterministic ``fake`` provider lets the whole
pipeline run and be tested offline with no Neo4j/Ollama. ``sentence-transformers``
can be added later behind the same :class:`Embedder` interface.
"""

from __future__ import annotations

from secbrn.config import Settings
from secbrn.providers.base import Embedder, LLM
from secbrn.providers.fake import FakeEmbedder, FakeLLM
from secbrn.providers.ollama import OllamaEmbedder, OllamaLLM

__all__ = [
    "Embedder",
    "LLM",
    "get_embedder",
    "get_extract_llm",
    "get_answer_llm",
    "get_rerank_llm",
]


def get_embedder(settings: Settings) -> Embedder:
    if settings.provider == "fake":
        return FakeEmbedder(dim=settings.embed_dim, model="fake-embed")
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embed_model,
        dim=settings.embed_dim,
        timeout=settings.ollama_embed_timeout,
        max_retries=settings.ollama_max_retries,
        backoff=settings.ollama_retry_backoff,
    )


def get_extract_llm(settings: Settings) -> LLM:
    if settings.provider == "fake":
        return FakeLLM(model="fake-extract")
    return OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.extract_model,
        timeout=settings.ollama_llm_timeout,
        max_retries=settings.ollama_max_retries,
        backoff=settings.ollama_retry_backoff,
    )


def get_answer_llm(settings: Settings) -> LLM:
    if settings.provider == "fake":
        return FakeLLM(model="fake-answer")
    return OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.answer_model,
        timeout=settings.ollama_llm_timeout,
        max_retries=settings.ollama_max_retries,
        backoff=settings.ollama_retry_backoff,
    )


def get_rerank_llm(settings: Settings) -> LLM:
    """LLM used for reranking + query expansion. Uses rerank_model if set, else answer_model."""
    if settings.provider == "fake":
        return FakeLLM(model="fake-rerank")
    return OllamaLLM(
        base_url=settings.ollama_base_url,
        model=settings.rerank_model or settings.answer_model,
        timeout=settings.ollama_llm_timeout,
        max_retries=settings.ollama_max_retries,
        backoff=settings.ollama_retry_backoff,
    )
