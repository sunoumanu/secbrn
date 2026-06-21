"""Reranking (Stage 7.5) — listwise LLM rerank of the fused top candidates.

A cross-encoder-style reorder using the local LLM: one call per query that ranks the
top-N passages by how well they answer the query. Falls back to the original order on
any parse failure, so it can never make retrieval *worse* than the input ordering in
a hard error case. Off unless SECBRN_RERANK.
"""

from __future__ import annotations

import json
import re

from secbrn.models import RetrievedChunk
from secbrn.providers.base import LLM

_SYSTEM = (
    "You are a search reranker. Given a query and numbered passages, return ONLY a JSON "
    "list of the passage numbers ordered best-first. Example: [3,1,2]."
)


class LLMReranker:
    def __init__(self, llm: LLM, snippet_chars: int = 400):
        self.llm = llm
        self.snippet_chars = snippet_chars

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        head = chunks[: max(0, top_n)]
        tail = chunks[max(0, top_n):]
        if len(head) < 2:
            return chunks
        listing = "\n".join(
            f"[{i + 1}] {c.text[: self.snippet_chars]}" for i, c in enumerate(head)
        )
        prompt = (
            f"QUERY: {query}\n\nPASSAGES:\n{listing}\n\n"
            "Return the passage numbers ordered best-first as a JSON list."
        )
        try:
            raw = self.llm.complete_json(prompt, system=_SYSTEM)
            order = _parse_order(raw, len(head))
        except Exception:
            order = []
        if not order:
            return chunks  # fallback: keep fused order

        # Reorder head by the model's ranking; assign descending scores so any later
        # stable sort keeps this order. Unreferenced passages keep their relative order.
        reordered: list[RetrievedChunk] = []
        seen = set()
        n = len(head)
        for rank, idx in enumerate(order):
            if 1 <= idx <= n and idx not in seen:
                seen.add(idx)
                c = head[idx - 1]
                c.score = float(n - rank)
                if "rerank" not in c.via:
                    c.via = f"{c.via}+rerank"
                reordered.append(c)
        for i, c in enumerate(head, start=1):
            if i not in seen:
                reordered.append(c)
        return reordered + tail


def _parse_order(raw: str, n: int) -> list[int]:
    raw = raw.strip()
    if "[" in raw and "]" in raw:
        frag = raw[raw.find("[") : raw.rfind("]") + 1]
        try:
            arr = json.loads(frag)
            return [int(x) for x in arr if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()]
        except Exception:
            pass
    # fallback: pull integers in order
    return [int(x) for x in re.findall(r"\d+", raw)][:n]
