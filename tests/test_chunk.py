"""Stage 3 — chunking tests (structure-aware spans)."""

from __future__ import annotations

from secbrn.ingest.chunk import chunk_document
from secbrn.models import Document


def _doc(source_type, text):
    return Document(id="d", source_type=source_type, uri="file://d", title="t", raw_text=text)


def test_pdf_pages_keep_page_spans():
    doc = _doc("pdf", "[[PAGE 1]]\nalpha beta\n[[PAGE 2]]\ngamma delta")
    chunks = chunk_document(doc, chunk_size=400, overlap=40)
    kinds = {c.span.kind for c in chunks}
    assert kinds == {"page"}
    pages = sorted(c.span.start for c in chunks)
    assert pages == [1, 2]


def test_transcript_keeps_turn_spans():
    text = "[[TURN 1]] Alice: hello there\n[[TURN 2]] Bob: hi back"
    doc = _doc("transcript", text)
    chunks = chunk_document(doc, chunk_size=400, overlap=40)
    assert {c.span.kind for c in chunks} == {"turn"}
    assert chunks[0].span.label == "Alice"


def test_markdown_sections_and_ids():
    text = "# Title\nintro\n## Section A\nbody a\n## Section B\nbody b"
    doc = _doc("markdown", text)
    chunks = chunk_document(doc, chunk_size=400, overlap=40)
    assert all(c.id.startswith("d:") for c in chunks)
    labels = {c.span.label for c in chunks}
    assert "Section A" in labels and "Section B" in labels


def test_long_text_windows_with_overlap():
    body = ". ".join(f"sentence number {i}" for i in range(200))
    doc = _doc("markdown", body)
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)  # windows bounded
