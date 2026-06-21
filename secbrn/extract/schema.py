"""The closed graph schema — the project's main lever against extraction noise.

Mirrors docs/SCHEMA.md. Keep these sets SMALL and CLOSED. Every label/relation
added multiplies the ways extraction can disagree with itself. Grow only when a
real query fails for lack of a type (see SCHEMA.md §6 on migrations).
"""

from __future__ import annotations

# ── Semantic node labels (LLM-extracted, closed set) ──────────────────────────
ENTITY_LABELS: tuple[str, ...] = (
    "Concept",
    "Tool",
    "Person",
    "Org",
    "Topic",
    "Event",
    "Place",
)

# ── Relationship types between entities (closed set) ──────────────────────────
RELATION_TYPES: tuple[str, ...] = (
    "RELATES_TO",     # generic association (fallback)
    "PART_OF",        # composition / hierarchy
    "USES",           # usage / dependency
    "IMPROVES",       # one enhances another
    "ALTERNATIVE_TO", # competing / substitute
    "AUTHORED_BY",    # authorship (Document → Person)
)

# ── Valid (subject_label, relation, object_label) triples ─────────────────────
# SchemaLLMPathExtractor (or our constrained extractor) drops anything off this
# list. "Entity" matches any semantic label (permissive fallback).
VALID_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("Tool", "ALTERNATIVE_TO", "Tool"),
    ("Tool", "USES", "Tool"),
    ("Person", "USES", "Tool"),
    ("Org", "USES", "Tool"),
    ("Concept", "IMPROVES", "Concept"),
    ("Concept", "PART_OF", "Topic"),
    ("Tool", "PART_OF", "Topic"),
    ("Concept", "RELATES_TO", "Concept"),
    ("Tool", "RELATES_TO", "Concept"),
    ("Person", "PART_OF", "Org"),
    ("Document", "AUTHORED_BY", "Person"),
    ("Entity", "RELATES_TO", "Entity"),  # permissive fallback
)

STRICT = True  # drop non-conforming triples rather than coercing them


def _matches(slot: str, label: str) -> bool:
    return slot == "Entity" or slot == label


def is_valid_triple(subject_label: str, relation: str, object_label: str) -> bool:
    """True if (subject_label, relation, object_label) is permitted by the schema."""
    if relation not in RELATION_TYPES:
        return False
    if subject_label not in ENTITY_LABELS and subject_label not in {"Document", "Entity"}:
        return False
    if object_label not in ENTITY_LABELS and object_label not in {"Entity"}:
        return False
    for s, r, o in VALID_TRIPLES:
        if r == relation and _matches(s, subject_label) and _matches(o, object_label):
            return True
    return not STRICT


def normalize_label(label: str) -> str:
    """Coerce a model-emitted label to the closed set; default to Concept."""
    label = (label or "").strip().capitalize()
    aliases = {
        "Software": "Tool",
        "Library": "Tool",
        "Product": "Tool",
        "Framework": "Tool",
        "Organization": "Org",
        "Company": "Org",
        "Subject": "Topic",
        "Idea": "Concept",
        "Location": "Place",
    }
    label = aliases.get(label, label)
    return label if label in ENTITY_LABELS else "Concept"


def schema_prompt_block() -> str:
    """Human-readable schema description injected into the extraction prompt."""
    triples = "\n".join(f"  ({s}) -[{r}]-> ({o})" for s, r, o in VALID_TRIPLES)
    return (
        "ALLOWED ENTITY LABELS:\n  " + ", ".join(ENTITY_LABELS) + "\n\n"
        "ALLOWED RELATION TYPES:\n  " + ", ".join(RELATION_TYPES) + "\n\n"
        "ALLOWED TRIPLES (subject_label -[relation]-> object_label):\n" + triples
    )
