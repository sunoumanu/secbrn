"""Healthcheck — Phase 0 exit criterion.

`python -m secbrn.healthcheck` confirms Neo4j + both models reachable and indexes
present. Returns non-zero on failure so it can gate CI / setup scripts.
"""

from __future__ import annotations

import sys

from secbrn.config import get_settings
from secbrn.graph import get_store
from secbrn.providers import get_answer_llm, get_embedder, get_extract_llm


def run() -> int:
    s = get_settings()
    ok = True
    lines: list[str] = [f"SecBrn healthcheck (provider={s.provider})"]

    # Neo4j / store
    store = get_store(s)
    try:
        store.ensure_schema()
        reachable = store.ping()
        lines.append(f"  [{'OK' if reachable else 'FAIL'}] graph store reachable")
        ok &= reachable
        idx = store.indexes_present()
        for name, present in idx.items():
            lines.append(f"  [{'OK' if present else 'FAIL'}] index {name}")
            ok &= present
    except Exception as e:  # pragma: no cover - depends on live infra
        lines.append(f"  [FAIL] graph store: {e}")
        ok = False
    finally:
        store.close()

    # Embeddings
    try:
        emb = get_embedder(s)
        v = emb.embed_one("healthcheck")
        good = len(v) == s.embed_dim
        lines.append(f"  [{'OK' if good else 'FAIL'}] embed model '{emb.model}' dim={len(v)}")
        ok &= good
    except Exception as e:  # pragma: no cover
        lines.append(f"  [FAIL] embed model: {e}")
        ok = False

    # LLMs
    for label, getter in (("extract", get_extract_llm), ("answer", get_answer_llm)):
        try:
            llm = getter(s)
            out = llm.complete("Reply with the single word: ok")
            good = bool(out.strip())
            lines.append(f"  [{'OK' if good else 'FAIL'}] {label} model '{llm.model}'")
            ok &= good
        except Exception as e:  # pragma: no cover
            lines.append(f"  [FAIL] {label} model: {e}")
            ok = False

    lines.append("RESULT: " + ("HEALTHY" if ok else "UNHEALTHY"))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run())
