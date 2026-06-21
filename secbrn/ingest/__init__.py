"""Ingestion subsystem: Stage 1 (load) → Stage 2 (normalize/dedupe) → Stage 3 (chunk)."""

from secbrn.ingest.chunk import chunk_document
from secbrn.ingest.loaders import (
    iter_folder,
    load_markdown,
    load_path,
    load_pdf,
    load_transcript,
    load_web,
)
from secbrn.ingest.normalize import NormalizeResult, normalize_and_dedupe, normalize_text

__all__ = [
    "load_path",
    "load_markdown",
    "load_pdf",
    "load_web",
    "load_transcript",
    "iter_folder",
    "normalize_and_dedupe",
    "normalize_text",
    "NormalizeResult",
    "chunk_document",
]
