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


def test_extractor_survives_malformed_json(monkeypatch):
    """Regression: a model returning entities/relations as bare strings (or a
    bare list / string instead of an object) must not raise
    ``AttributeError: 'str' object has no attribute 'get'`` — the chunk should
    simply yield an empty/partial extraction instead of failing."""
    import json

    malformed_payloads = [
        json.dumps({"entities": ["Raskolnikov", "Sonia"], "relations": []}),  # bare strings
        json.dumps(["Raskolnikov", "Sonia"]),                                  # bare list
        json.dumps("Raskolnikov"),                                            # bare string
        json.dumps({"entities": "Raskolnikov", "relations": "x"}),            # string values
        json.dumps({
            "entities": [{"name": "Alice", "label": "Person"}, "junk", 42, None],
            "relations": [{"subject": "Alice", "relation": "RELATES_TO", "object": "Bob"}],
        }),                                                                   # mixed garbage
    ]

    class ScriptableLLM:
        model = "scriptable"

        def __init__(self, payload):
            self._payload = payload

        def complete_json(self, prompt, *, system=None):
            return self._payload

        def complete(self, *a, **k):
            return ""

    for payload in malformed_payloads:
        ex = extract_chunk("text", ScriptableLLM(payload))  # must not raise
        # well-formed items inside otherwise-garbage payloads still survive
        assert all(isinstance(e.name, str) and e.name for e in ex.entities)
