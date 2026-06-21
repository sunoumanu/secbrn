"""Stage 5 — Schema-constrained KG extraction.

A pragmatic, framework-free stand-in for LlamaIndex ``SchemaLLMPathExtractor``: the
LLM is prompted with the closed schema and asked for JSON; we then *enforce* the
schema ourselves (drop entities with bad labels, drop triples not in VALID_TRIPLES).
This keeps the noise-control guarantee regardless of model compliance, and the same
``Extraction`` output can later be produced by the real SchemaLLMPathExtractor
without changing downstream code.
"""

from __future__ import annotations

import json

from secbrn.extract import schema as S
from secbrn.models import ExtractedEntity, ExtractedRelation, Extraction
from secbrn.providers.base import LLM

_SYSTEM = (
    "You are a careful knowledge-graph extractor. You ONLY emit entities and "
    "relationships permitted by the given schema. If unsure, omit. Output strict JSON."
)


def _prompt(text: str) -> str:
    return (
        f"{S.schema_prompt_block()}\n\n"
        "Extract entities and relationships from the TEXT below.\n"
        "Return JSON: {\"entities\":[{\"name\":..,\"label\":..,\"aliases\":[..]}],"
        "\"relations\":[{\"subject\":..,\"relation\":..,\"object\":..}]}\n"
        "Rules: names are canonical surface forms; label MUST be one of the allowed "
        "labels; relation MUST be one of the allowed types; only emit a relation if its "
        "(subject_label, relation, object_label) is an allowed triple.\n\n"
        f"TEXT:\n{text}"
    )


def extract_chunk(text: str, llm: LLM) -> Extraction:
    """Run the LLM then validate against the closed schema (defensive)."""
    raw = llm.complete_json(_prompt(text), system=_SYSTEM)
    data = _safe_json(raw)
    return _validate(data)


def _safe_json(raw: str) -> dict:
    raw = raw.strip()
    # tolerate models that wrap JSON in prose / code fences
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip() if raw.count("```") >= 2 else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except Exception:
        return {"entities": [], "relations": []}


def _validate(data: dict) -> Extraction:
    label_by_name: dict[str, str] = {}
    entities: list[ExtractedEntity] = []
    for e in data.get("entities", []) or []:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        label = S.normalize_label(str(e.get("label", "Concept")))
        aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()]
        label_by_name[name] = label
        entities.append(ExtractedEntity(name=name, label=label, aliases=aliases,
                                        summary=str(e.get("summary", "")).strip()))

    relations: list[ExtractedRelation] = []
    for r in data.get("relations", []) or []:
        subj = str(r.get("subject", "")).strip()
        obj = str(r.get("object", "")).strip()
        rel = str(r.get("relation", "")).strip().upper()
        if not subj or not obj or subj == obj:
            continue
        # entities referenced by a relation must have been declared
        sl = label_by_name.get(subj)
        ol = label_by_name.get(obj)
        if sl is None or ol is None:
            continue
        if S.is_valid_triple(sl, rel, ol):
            relations.append(ExtractedRelation(subject=subj, relation=rel, object=obj))
    return Extraction(entities=entities, relations=relations)
