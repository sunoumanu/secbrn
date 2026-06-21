"""Stage 2 — Normalize & deduplicate sources.

Strip boilerplate, normalize whitespace/encoding, compute a content hash for
idempotent ingestion, and detect updates (same uri, new hash → new version).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from secbrn.graph.base import GraphStore
from secbrn.models import Document, content_hash

_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of spaces/tabs but keep our [[PAGE n]] / [[TURN n]] markers intact
    lines = [_MULTISPACE.sub(" ", ln).rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


@dataclass
class NormalizeResult:
    document: Document
    status: str  # "new" | "duplicate" | "updated"


def normalize_and_dedupe(doc: Document, store: GraphStore) -> NormalizeResult:
    """Normalize, hash, and decide new/duplicate/updated against the store."""
    doc.raw_text = normalize_text(doc.raw_text)
    doc.content_hash = content_hash(doc.raw_text)

    # Same exact content already present anywhere → skip (idempotent).
    if store.content_hash_exists(doc.content_hash):
        return NormalizeResult(doc, "duplicate")

    # Same source URI but different content → new version, re-extract.
    existing = store.get_document_by_uri(doc.uri)
    if existing is not None:
        doc.id = existing.id          # keep stable id across versions
        doc.version = existing.version + 1
        return NormalizeResult(doc, "updated")

    return NormalizeResult(doc, "new")
