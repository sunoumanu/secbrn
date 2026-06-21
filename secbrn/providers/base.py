"""Provider interfaces. Keep these tiny so backends are easy to swap."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text. Length-stable, dim == self.dim."""
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


@runtime_checkable
class LLM(Protocol):
    model: str

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        """Single-shot completion. Deterministic by default (temperature 0)."""
        ...

    def complete_json(self, prompt: str, *, system: str | None = None) -> str:
        """Completion requested in JSON mode where the backend supports it."""
        ...
