"""Shared test fixtures.

Everything runs offline: in-memory graph store, fake embedder, and a *scripted* LLM
that emits schema-correct extractions for the known fixture vocabulary. The scripted
LLM lets the end-to-end test assert a real multi-hop connection rather than depend on
heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from secbrn.config import Settings
from secbrn.graph.memory import InMemoryStore
from secbrn.pipeline import Brain
from secbrn.providers.fake import FakeEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedLLM:
    """Deterministic, schema-aware stand-in for the extraction + answer LLM."""

    model = "scripted"

    # keyword → (label) for entity typing
    _ENTITIES = {
        "Retrieval": "Concept",
        "Reranking": "Concept",
        "Cross-Encoder": "Tool",
        "Neo4j": "Tool",
        "pgvector": "Tool",
        "Ollama": "Tool",
        "LlamaIndex": "Tool",
        "RAG": "Topic",
        "GraphRAG": "Topic",
        "Graph expansion": "Concept",
    }

    def complete_json(self, prompt: str, *, system: str | None = None) -> str:
        import re as _re

        text = prompt.split("TEXT:", 1)[-1]
        # word-boundary match so e.g. "RAG" is not found inside "storage"
        found = [
            name for name in self._ENTITIES
            if _re.search(r"\b" + _re.escape(name) + r"\b", text, _re.IGNORECASE)
        ]
        entities = [{"name": n, "label": self._ENTITIES[n], "aliases": []} for n in found]
        rels = []

        def has(*names):
            return all(n in found for n in names)

        if has("Reranking", "Retrieval"):
            rels.append({"subject": "Reranking", "relation": "IMPROVES", "object": "Retrieval"})
        if has("Cross-Encoder", "Reranking"):
            rels.append({"subject": "Cross-Encoder", "relation": "RELATES_TO", "object": "Reranking"})
        if has("Reranking", "RAG"):
            rels.append({"subject": "Reranking", "relation": "PART_OF", "object": "RAG"})
        if has("pgvector", "Neo4j"):
            rels.append({"subject": "pgvector", "relation": "ALTERNATIVE_TO", "object": "Neo4j"})
        if has("LlamaIndex", "Ollama"):
            rels.append({"subject": "LlamaIndex", "relation": "USES", "object": "Ollama"})
        if has("Graph expansion", "Neo4j"):
            rels.append({"subject": "Graph expansion", "relation": "RELATES_TO", "object": "Neo4j"})
        return json.dumps({"entities": entities, "relations": rels})

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        import re

        markers = sorted(set(re.findall(r"\[(\d+)\]", prompt)), key=int)
        cited = " ".join(f"[{m}]" for m in markers[:3]) or "[1]"
        return (
            "Reranking improves retrieval, and a cross-encoder is the reranker used in RAG "
            f"pipelines. These facts connect across the notes {cited}."
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        provider="fake",
        embed_dim=64,
        chunk_size=400,
        chunk_overlap=60,
        retrieve_top_k=6,
        retrieve_hops=2,
    )


@pytest.fixture
def brain(settings) -> Brain:
    llm = ScriptedLLM()
    b = Brain(
        settings=settings,
        store=InMemoryStore(),
        embedder=FakeEmbedder(dim=settings.embed_dim),
        extract_llm=llm,
        answer_llm=llm,
    )
    yield b
    b.close()
