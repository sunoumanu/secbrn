"""Query expansion — augment the query with LLM-generated keywords/phrasings.

Helps paraphrase queries match passages that use different words. We append the
expansion terms to the query used for vector + full-text search, but keep the
ORIGINAL query for entity seeding and title matching (so expansion can't pollute
those exact-term signals). One LLM call per query; off unless SECBRN_QUERY_EXPANSION.
"""

from __future__ import annotations

import re

from secbrn.providers.base import LLM

_SPLIT = re.compile(r"[,\n;]+")
_SYSTEM = "You expand search queries. Output only the terms, comma-separated, no prose."


def expand_query(query: str, llm: LLM, n_terms: int = 6) -> str:
    """Return the query augmented with up to ``n_terms`` expansion keywords."""
    prompt = (
        f"Give {n_terms} alternative keywords or short phrasings a document might use to "
        f"answer this query. Comma-separated, no numbering, no explanation.\n\nQUERY: {query}"
    )
    try:
        out = llm.complete(prompt, system=_SYSTEM)
    except Exception:
        return query
    terms: list[str] = []
    qlow = query.lower()
    for t in _SPLIT.split(out):
        t = t.strip().strip("-•*\"'").strip()
        if t and t.lower() not in qlow and t.lower() not in (x.lower() for x in terms):
            terms.append(t)
        if len(terms) >= n_terms:
            break
    return (query + " " + " ".join(terms)).strip() if terms else query
