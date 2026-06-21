"""Deterministic offline providers for tests and demos (no Ollama needed).

The fake embedder is a hashed bag-of-words projection: similar text → similar
vectors, so vector search behaves sensibly in tests without a real model. The fake
LLM does light rule-based extraction and template answering so the *plumbing* of
Stages 5/8 is exercised end-to-end.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class FakeEmbedder:
    def __init__(self, dim: int = 768, model: str = "fake-embed"):
        self.dim = dim
        self.model = model

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class FakeLLM:
    """Rule-based stand-in.

    For extraction prompts (detected by a marker line) it returns schema-shaped JSON
    derived from simple capitalized-term heuristics. For answer prompts it stitches a
    grounded-looking response that references the provided citation markers.
    """

    def __init__(self, model: str = "fake-llm"):
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        if "ANSWER THE QUESTION" in prompt:
            return self._fake_answer(prompt)
        return self._fake_extract_json(prompt)

    def complete_json(self, prompt: str, *, system: str | None = None) -> str:
        return self._fake_extract_json(prompt)

    # -- extraction --------------------------------------------------------------
    def _fake_extract_json(self, prompt: str) -> str:
        # Pull the chunk text out of the prompt (everything after the TEXT: marker).
        text = prompt.split("TEXT:", 1)[-1]
        caps = re.findall(r"\b([A-Z][A-Za-z0-9]+(?:\s[A-Z][A-Za-z0-9]+)?)\b", text)
        seen: list[str] = []
        for c in caps:
            if c not in seen and c.lower() not in {"the", "this", "a", "an"}:
                seen.append(c)
        entities = [{"name": c, "label": "Concept", "aliases": []} for c in seen[:6]]
        relations = []
        for i in range(len(entities) - 1):
            relations.append(
                {"subject": entities[i]["name"], "relation": "RELATES_TO", "object": entities[i + 1]["name"]}
            )
        return json.dumps({"entities": entities, "relations": relations})

    # -- answering ---------------------------------------------------------------
    def _fake_answer(self, prompt: str) -> str:
        markers = re.findall(r"\[(\d+)\]", prompt)
        uniq = sorted(set(markers), key=int)
        cited = " ".join(f"[{m}]" for m in uniq[:3]) or "[1]"
        return (
            "Based on the retrieved notes and the connecting relationships in the graph, "
            f"the relevant facts link together as described in the sources {cited}."
        )
