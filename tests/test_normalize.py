"""Stage 2 — normalize + dedupe + versioning tests."""

from __future__ import annotations

from secbrn.graph.memory import InMemoryStore
from secbrn.ingest.normalize import normalize_and_dedupe, normalize_text
from secbrn.models import Document


def test_normalize_collapses_whitespace():
    assert normalize_text("a   b\r\n\n\n\nc  ") == "a b\n\nc"


def _doc(uri, text):
    return Document(id="x", source_type="markdown", uri=uri, title="t", raw_text=text)


def test_dedupe_skips_identical_content():
    store = InMemoryStore()
    r1 = normalize_and_dedupe(_doc("file://a", "hello world"), store)
    assert r1.status == "new"
    store.upsert_document(r1.document)
    r2 = normalize_and_dedupe(_doc("file://b", "hello world"), store)
    assert r2.status == "duplicate"  # same content hash, different uri


def test_update_detection_versions():
    store = InMemoryStore()
    r1 = normalize_and_dedupe(_doc("file://a", "v one"), store)
    store.upsert_document(r1.document)
    r2 = normalize_and_dedupe(_doc("file://a", "v two changed"), store)
    assert r2.status == "updated"
    assert r2.document.version == 2
    assert r2.document.id == r1.document.id  # stable id across versions
