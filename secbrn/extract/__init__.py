"""Extraction subsystem (Stage 5) + the closed schema (docs/SCHEMA.md)."""

from secbrn.extract import schema
from secbrn.extract.extractor import extract_chunk

__all__ = ["schema", "extract_chunk"]
