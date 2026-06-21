"""Stage 3 — Chunk. Structure-aware splitting with stable IDs + citation spans.

Prefers natural boundaries: PDF pages (``[[PAGE n]]``), transcript turns
(``[[TURN n]]``), Markdown headings. Falls back to a sentence-window splitter sized
by config. Each chunk records its document, position, and span for citations.
"""

from __future__ import annotations

import re

from secbrn.models import Chunk, Document, Span

_PAGE = re.compile(r"\[\[PAGE (\d+)\]\]")
_TURN = re.compile(r"\[\[TURN (\d+)\]\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_document(doc: Document, *, chunk_size: int, overlap: int) -> list[Chunk]:
    if doc.source_type == "pdf" and _PAGE.search(doc.raw_text):
        segments = _split_pdf_pages(doc.raw_text)
        kind = "page"
    elif doc.source_type == "transcript" and _TURN.search(doc.raw_text):
        segments = _split_transcript_turns(doc.raw_text)
        kind = "turn"
    else:
        segments = _split_markdown(doc.raw_text)
        kind = "section"

    chunks: list[Chunk] = []
    pos = 0
    for seg_label, seg_index, seg_text in segments:
        for piece, (lo, hi) in _window(seg_text, chunk_size, overlap):
            if not piece.strip():
                continue
            if kind in ("page", "turn"):
                span = Span(kind=kind, start=seg_index, end=seg_index, label=seg_label)
            else:
                span = Span(kind="line", start=lo, end=hi, label=seg_label)
            chunks.append(
                Chunk(
                    id=f"{doc.id}:{pos}",
                    document_id=doc.id,
                    position=pos,
                    text=piece.strip(),
                    span=span,
                )
            )
            pos += 1
    return chunks


def _split_pdf_pages(text: str) -> list[tuple[str | None, int, str]]:
    out: list[tuple[str | None, int, str]] = []
    parts = re.split(_PAGE, text)
    # parts: ['', '1', 'page1 text', '2', 'page2 text', ...]
    for i in range(1, len(parts), 2):
        page_no = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        out.append((f"p.{page_no}", page_no, body))
    return out


def _split_transcript_turns(text: str) -> list[tuple[str | None, int, str]]:
    out: list[tuple[str | None, int, str]] = []
    parts = re.split(_TURN, text)
    for i in range(1, len(parts), 2):
        turn_no = int(parts[i])
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        speaker = body.split(":", 1)[0].strip() if ":" in body[:40] else None
        out.append((speaker, turn_no, body))
    return out


def _split_markdown(text: str) -> list[tuple[str | None, int, str]]:
    """Split on headings; track starting line number for each section span."""
    lines = text.split("\n")
    sections: list[tuple[str | None, int, list[str]]] = []
    cur_label: str | None = None
    cur_start = 1
    buf: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if re.match(r"^#{1,6}\s", line):
            if buf:
                sections.append((cur_label, cur_start, buf))
            cur_label = re.sub(r"^#{1,6}\s", "", line).strip()
            cur_start = idx
            buf = [line]
        else:
            buf.append(line)
    if buf:
        sections.append((cur_label, cur_start, buf))
    if not sections:
        sections = [(None, 1, lines)]
    return [(lbl, start, "\n".join(b)) for lbl, start, b in sections]


def _window(text: str, size: int, overlap: int):
    """Yield (piece, (start_offset, end_offset)) windows over sentence boundaries."""
    text = text.strip()
    if len(text) <= size:
        yield text, (0, len(text))
        return
    sentences = _SENT.split(text)
    buf = ""
    start = 0
    cursor = 0
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > size:
            yield buf, (start, cursor)
            # start next window with an overlap tail
            tail = buf[-overlap:] if overlap else ""
            start = cursor - len(tail)
            buf = (tail + " " + sent).strip()
        else:
            buf = (buf + " " + sent).strip() if buf else sent
        cursor += len(sent) + 1
    if buf:
        yield buf, (start, cursor)
