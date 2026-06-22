"""Backfill correct line-number citation spans onto already-ingested chunks.

Chunks ingested before the span fix stored *character offsets* under a "line" span
(so `secbrn ask` printed nonsense like "line 845254"). Re-running the deterministic
chunker on the original source and writing only `c.span` repairs those citations
WITHOUT re-embedding or re-extracting (no Ollama calls, seconds not hours).

Safety: a document is only touched when the freshly normalized source still hashes
to the value stored at ingest time, so a changed file can never corrupt spans.

Usage:
    python scripts/backfill_spans.py <path-you-ingested>      # e.g. my-notes\
    python scripts/backfill_spans.py <path> --dry-run         # report only
"""

from __future__ import annotations

import sys
from pathlib import Path

from secbrn.config import get_settings
from secbrn.ingest.chunk import chunk_document
from secbrn.ingest.loaders import iter_folder, load_path
from secbrn.ingest.normalize import normalize_text
from secbrn.models import content_hash
from secbrn.pipeline import Brain


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry = "--dry-run" in sys.argv
    if not args:
        print(__doc__)
        return 2
    root = Path(args[0])
    files = iter_folder(root) if root.is_dir() else [root]

    s = get_settings()
    brain = Brain.from_env()
    store = brain.store
    if not hasattr(store, "update_chunk_span"):
        print("This store backend has no update_chunk_span (in-memory isn't persisted).")
        brain.close()
        return 1

    docs_done = docs_skipped = spans_updated = spans_missing = 0
    try:
        for f in files:
            try:
                doc = load_path(f)
            except Exception as e:
                print(f"  skip {f}: cannot load ({type(e).__name__}: {e})")
                continue

            doc.raw_text = normalize_text(doc.raw_text)
            existing = store.get_document_by_uri(doc.uri)
            if existing is None:
                print(f"  skip {doc.title}: not in graph (uri not found)")
                docs_skipped += 1
                continue
            if content_hash(doc.raw_text) != existing.content_hash:
                print(f"  skip {doc.title}: source changed since ingest (hash mismatch) — re-ingest instead")
                docs_skipped += 1
                continue

            # Reuse the stored id so chunk ids ({id}:{pos}) line up with the graph.
            doc.id = existing.id
            chunks = chunk_document(doc, chunk_size=s.chunk_size, overlap=s.chunk_overlap)
            updated = missing = 0
            for c in chunks:
                if dry:
                    continue
                if store.update_chunk_span(c.id, c.span):
                    updated += 1
                else:
                    missing += 1
            docs_done += 1
            spans_updated += updated
            spans_missing += missing
            tag = "(dry-run) " if dry else ""
            print(f"  {tag}{doc.title}: {len(chunks)} chunks, "
                  f"updated={updated} missing={missing}")
    finally:
        brain.close()

    print(f"\nDocuments repaired={docs_done} skipped={docs_skipped} "
          f"spans_updated={spans_updated} spans_missing={spans_missing}")
    if dry:
        print("Dry run — nothing was written. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
