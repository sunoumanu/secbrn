"""Stage 5 — schema validation + constrained extraction tests."""

from __future__ import annotations

from secbrn.extract import schema as S
from secbrn.extract.extractor import extract_chunk
from tests.conftest import ScriptedLLM


def test_valid_triples():
    assert S.is_valid_triple("Concept", "IMPROVES", "Concept")
    assert S.is_valid_triple("Tool", "ALTERNATIVE_TO", "Tool")
    assert S.is_valid_triple("Entity", "RELATES_TO", "Entity")


def test_invalid_triples_rejected():
    assert not S.is_valid_triple("Person", "IMPROVES", "Tool")   # not allowed
    assert not S.is_valid_triple("Concept", "BOGUS_REL", "Concept")


def test_label_normalization():
    assert S.normalize_label("library") == "Tool"
    assert S.normalize_label("company") == "Org"
    assert S.normalize_label("nonsense") == "Concept"  # default


def test_extractor_drops_nonconforming(monkeypatch):
    # an LLM that emits an illegal triple + a bad label
    import json

    class BadLLM:
        model = "bad"

        def complete_json(self, prompt, *, system=None):
            return json.dumps(
                {
                    "entities": [
                        {"name": "Alice", "label": "Person"},
                        {"name": "Neo4j", "label": "Tool"},
                    ],
                    "relations": [
                        {"subject": "Alice", "relation": "IMPROVES", "object": "Neo4j"},  # invalid
                    ],
                }
            )

        def complete(self, *a, **k):
            return ""

    ex = extract_chunk("text", BadLLM())
    assert {e.name for e in ex.entities} == {"Alice", "Neo4j"}
    assert ex.relations == []  # illegal triple dropped


def test_scripted_extraction_conforms():
    ex = extract_chunk("TEXT: Reranking improves Retrieval", ScriptedLLM())
    assert any(r.relation == "IMPROVES" for r in ex.relations)
