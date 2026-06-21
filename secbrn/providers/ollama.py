"""Ollama-backed providers (default). Talks to a local Ollama HTTP server.

Hardened for long-running ingests of large documents: timeouts are configurable and
transient failures (read timeouts, connection drops) are retried with exponential
backoff before the error propagates.
"""

from __future__ import annotations

import time

import httpx

# Transient errors worth retrying (vs. e.g. HTTP 4xx which won't get better).
_TRANSIENT = (httpx.TimeoutException, httpx.TransportError, httpx.RemoteProtocolError)


class _OllamaClient:
    def __init__(self, base_url: str, timeout: float, max_retries: int, backoff: float):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(1, max_retries)
        self.backoff = backoff
        # trust_env=False: Ollama is a localhost service; a system HTTP/SOCKS proxy
        # should not intercept it (and avoids needing optional proxy extras).
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, trust_env=False)

    def post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except _TRANSIENT as e:
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (2 ** attempt))
            except httpx.HTTPStatusError as e:
                # Server-side 5xx can be transient; client 4xx is not.
                last = e
                if e.response is not None and e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(self.backoff * (2 ** attempt))
                else:
                    raise
        assert last is not None
        raise last


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, dim: int, *, timeout: float = 180.0,
                 max_retries: int = 3, backoff: float = 2.0):
        self.model = model
        self.dim = dim
        self._c = _OllamaClient(base_url, timeout, max_retries, backoff)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        data = self._c.post("/api/embeddings", {"model": self.model, "prompt": text})
        vec = data["embedding"]
        if len(vec) != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: model '{self.model}' returned {len(vec)}, "
                f"config SECBRN_EMBED_DIM={self.dim}. Update the vector index + config."
            )
        return vec


class OllamaLLM:
    def __init__(self, base_url: str, model: str, *, timeout: float = 600.0,
                 max_retries: int = 3, backoff: float = 2.0):
        self.model = model
        self._c = _OllamaClient(base_url, timeout, max_retries, backoff)

    def complete(self, prompt: str, *, system: str | None = None, temperature: float = 0.0) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        return self._c.post("/api/generate", payload).get("response", "")

    def complete_json(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        if system:
            payload["system"] = system
        return self._c.post("/api/generate", payload).get("response", "")
