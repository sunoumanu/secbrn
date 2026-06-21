"""Stage 8 — Grounded answer synthesis with inline citations.

Assembles a prompt giving the LLM BOTH the retrieved chunks AND a serialized view of
the relevant subgraph (entities + typed edges). The graph structure is what lets the
model reason across documents rather than summarize one. Every chunk is numbered so
the model can cite ``[n]``; we flag answers that contain no citation.
"""

from __future__ import annotations

import re

from secbrn.models import Answer, Citation, ContextBundle
from secbrn.providers.base import LLM

_SYSTEM = (
    "You are a precise assistant answering ONLY from the provided context. "
    "Cite every claim with the bracketed source number, e.g. [1]. If the context "
    "does not contain the answer, say so. Use the RELATIONSHIPS to explain how facts "
    "connect across sources."
)


def _serialize_subgraph(bundle: ContextBundle, limit: int = 40) -> str:
    if not bundle.edges:
        return "(no relevant relationships found)"
    lines = [f"{e.subject} -[{e.relation}]-> {e.object}" for e in bundle.edges[:limit]]
    return "\n".join(lines)


def _build_prompt(question: str, bundle: ContextBundle) -> tuple[str, list[Citation]]:
    citations: list[Citation] = []
    blocks: list[str] = []
    for i, rc in enumerate(bundle.chunks, start=1):
        marker = f"[{i}]"
        citations.append(
            Citation(marker=marker, document_title=rc.document_title, uri=rc.uri, span=rc.span.cite())
        )
        blocks.append(f"{marker} {rc.document_title} ({rc.span.cite()}):\n{rc.text}")
    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    prompt = (
        "ANSWER THE QUESTION using only the SOURCES and RELATIONSHIPS below.\n\n"
        f"QUESTION: {question}\n\n"
        f"SOURCES:\n{context}\n\n"
        f"RELATIONSHIPS (knowledge graph):\n{_serialize_subgraph(bundle)}\n\n"
        "Write a concise answer. Cite sources inline as [n]. "
        "Explain connections using the relationships where relevant."
    )
    return prompt, citations


def synthesize(question: str, bundle: ContextBundle, llm: LLM) -> Answer:
    prompt, citations = _build_prompt(question, bundle)
    text = llm.complete(prompt, system=_SYSTEM).strip()
    used = set(re.findall(r"\[(\d+)\]", text))
    uncited = len(used) == 0 and bool(bundle.chunks)
    # keep only citations actually referenced (plus all if model under-cited)
    kept = [c for c in citations if c.marker.strip("[]") in used] or citations
    return Answer(text=text, citations=kept, bundle=bundle, uncited=uncited)
