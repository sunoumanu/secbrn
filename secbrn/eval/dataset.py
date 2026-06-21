"""Gold-set schema + loader for the eval harness.

A gold set is a JSON (or YAML, if PyYAML is installed) file with three optional
sections — evaluate any subset you have labels for:

{
  "corpus": "tests/fixtures",          // folder ingested before retrieval eval
  "retrieval": [
    {"query": "...", "relevant": ["Doc Title A", "https://uri/b"]}
  ],
  "extraction": [
    {"text": "...",
     "entities": [["Reranking","Concept"]],
     "triples":  [["Reranking","IMPROVES","Retrieval"]]}
  ],
  "resolution": [
    {"entities": [["pgvector","Tool"],["PGVector","Tool"],["Neo4j","Tool"]],
     "should_merge":     [["pgvector","PGVector"]],
     "should_not_merge": [["pgvector","Neo4j"]]}
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrievalCase:
    query: str
    relevant: list[str]  # document titles or URIs considered relevant


@dataclass
class ExtractionCase:
    text: str
    entities: list[tuple[str, str]] = field(default_factory=list)   # (name, label)
    triples: list[tuple[str, str, str]] = field(default_factory=list)  # (s, rel, o)


@dataclass
class ResolutionCase:
    entities: list[tuple[str, str]] = field(default_factory=list)   # (name, label)
    should_merge: list[tuple[str, str]] = field(default_factory=list)
    should_not_merge: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class GoldSet:
    corpus: str | None = None
    retrieval: list[RetrievalCase] = field(default_factory=list)
    extraction: list[ExtractionCase] = field(default_factory=list)
    resolution: list[ResolutionCase] = field(default_factory=list)
    base_dir: Path = field(default=Path("."))

    def corpus_path(self) -> Path | None:
        if not self.corpus:
            return None
        p = Path(self.corpus)
        return p if p.is_absolute() else (self.base_dir / p)


def _load_raw(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PyYAML not installed; use a .json gold set or `pip install pyyaml`.") from e
        return yaml.safe_load(text)
    return json.loads(text)


def load_goldset(path: str | Path) -> GoldSet:
    path = Path(path)
    raw = _load_raw(path)
    retrieval = [RetrievalCase(c["query"], list(c.get("relevant", []))) for c in raw.get("retrieval", [])]
    extraction = [
        ExtractionCase(
            text=c["text"],
            entities=[tuple(e) for e in c.get("entities", [])],
            triples=[tuple(t) for t in c.get("triples", [])],
        )
        for c in raw.get("extraction", [])
    ]
    resolution = [
        ResolutionCase(
            entities=[tuple(e) for e in c.get("entities", [])],
            should_merge=[tuple(p) for p in c.get("should_merge", [])],
            should_not_merge=[tuple(p) for p in c.get("should_not_merge", [])],
        )
        for c in raw.get("resolution", [])
    ]
    return GoldSet(
        corpus=raw.get("corpus"),
        retrieval=retrieval,
        extraction=extraction,
        resolution=resolution,
        base_dir=path.parent,
    )
