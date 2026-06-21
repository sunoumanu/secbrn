"""Stage 6 — Entity resolution / dedup / canonicalization.

The second core failure mode is fragmented duplicates (``pgvector`` / ``Pgvector`` /
``PGVector`` as three nodes). This dedicated pass:

  1. Blocking — group candidates cheaply (same label + similar normalized name).
  2. Similarity — string distance (normalized + edit) AND embedding similarity of
     entity context.
  3. Merge — collapse into one canonical node, keep variants as aliases, rewire edges.
  4. Canonicalize — pick the canonical surface form (most-mentioned, or alias seed).
  5. (Optional) LLM adjudication for pairs within a margin band.

Thresholds are config, not hardcoded. Merges are reversible (store records history).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from secbrn.config import Settings
from secbrn.graph.base import GraphStore, StoredEntity, cosine
from secbrn.providers.base import LLM

_NONALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    return _NONALNUM.sub("", name.lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def string_similarity(a: str, b: str) -> float:
    """1.0 == identical normalized forms; scaled edit distance otherwise."""
    na, nb = normalize_name(a), normalize_name(b)
    if na == nb:
        return 1.0
    dist = _levenshtein(na, nb)
    return 1.0 - dist / max(len(na), len(nb), 1)


@dataclass
class MergeDecision:
    canonical: str
    duplicate: str
    string_score: float
    embed_score: float | None
    reason: str  # "alias_seed" | "string" | "string+embed" | "llm"


def load_alias_seeds(settings: Settings) -> dict[str, str]:
    """Map of surface form → canonical, short-circuiting fuzzy logic for known cases."""
    seeds = {
        "pgvector": "pgvector",
        "Pgvector": "pgvector",
        "PGVector": "pgvector",
        "neo4j": "Neo4j",
        "Neo4J": "Neo4j",
        "NEO4J": "Neo4j",
        "ollama": "Ollama",
        "llamaindex": "LlamaIndex",
        "Llama-Index": "LlamaIndex",
    }
    if settings.alias_seed_path:
        p = Path(settings.alias_seed_path)
        if p.exists():
            seeds.update(json.loads(p.read_text(encoding="utf-8")))
    return seeds


class EntityResolver:
    def __init__(self, settings: Settings, store: GraphStore, llm: LLM | None = None):
        self.s = settings
        self.store = store
        self.llm = llm
        self.seeds = load_alias_seeds(settings)

    # ── blocking ────────────────────────────────────────────────────────────────
    def _blocks(self, entities: list[StoredEntity]) -> list[list[StoredEntity]]:
        """Group by (label, normalized-name prefix) to avoid O(n^2) global comparison."""
        buckets: dict[tuple[str, str], list[StoredEntity]] = {}
        for e in entities:
            key = (e.label, normalize_name(e.name)[:4])
            buckets.setdefault(key, []).append(e)
        # also block purely by label so cross-prefix near-dupes still compare
        by_label: dict[str, list[StoredEntity]] = {}
        for e in entities:
            by_label.setdefault(e.label, []).append(e)
        blocks = [b for b in buckets.values() if len(b) > 1]
        blocks += [b for b in by_label.values() if len(b) > 1]
        return blocks

    # ── planning ────────────────────────────────────────────────────────────────
    def plan(self) -> list[MergeDecision]:
        entities = self.store.all_entities()
        decisions: list[MergeDecision] = []
        seen_pairs: set[tuple[str, str]] = set()

        # 1) alias-seed canonicalization
        canon_map: dict[str, str] = {}
        for e in entities:
            target = self.seeds.get(e.name) or self.seeds.get(normalize_name(e.name))
            if target and target != e.name:
                canon_map[e.name] = target
        for dup, canon in canon_map.items():
            decisions.append(MergeDecision(canon, dup, 1.0, None, "alias_seed"))

        # 2) fuzzy within blocks
        emb_cache: dict[str, list[float] | None] = {}

        def emb(name: str):
            if name not in emb_cache:
                emb_cache[name] = self.store.entity_context_embedding(name)
            return emb_cache[name]

        for block in self._blocks(entities):
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    a, b = block[i], block[j]
                    if a.name == b.name:
                        continue
                    pair = tuple(sorted((a.name, b.name)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    ss = string_similarity(a.name, b.name)
                    wd = _levenshtein(normalize_name(a.name), normalize_name(b.name))
                    if ss < self.s.res_similarity_threshold and wd > self.s.res_word_distance:
                        continue
                    ea, eb = emb(a.name), emb(b.name)
                    es = cosine(ea, eb) if ea and eb else None
                    canon, dup = self._pick_canonical(a, b)
                    # strong string OR (decent string AND strong embed)
                    if ss >= self.s.res_similarity_threshold or wd <= self.s.res_word_distance:
                        if es is None or es >= self.s.res_embed_cutoff - self.s.res_llm_margin:
                            reason = "string" if es is None else "string+embed"
                            # ambiguous band → optional LLM tie-break
                            if (
                                es is not None
                                and abs(es - self.s.res_embed_cutoff) <= self.s.res_llm_margin
                                and self.llm is not None
                            ):
                                if not self._llm_same(canon, dup):
                                    continue
                                reason = "llm"
                            decisions.append(MergeDecision(canon, dup, ss, es, reason))
        return _dedup_decisions(decisions)

    def _pick_canonical(self, a: StoredEntity, b: StoredEntity) -> tuple[str, str]:
        # alias seed wins
        for e in (a, b):
            if e.name in self.seeds.values():
                other = b if e is a else a
                return e.name, other.name
        # else most-mentioned, tie-break by shorter/cleaner name
        if a.mention_count != b.mention_count:
            hi, lo = (a, b) if a.mention_count > b.mention_count else (b, a)
        else:
            hi, lo = (a, b) if len(a.name) <= len(b.name) else (b, a)
        return hi.name, lo.name

    def _llm_same(self, a: str, b: str) -> bool:
        out = self.llm.complete(
            f"Are '{a}' and '{b}' the same real-world entity? Answer yes or no.",
            temperature=0.0,
        ).strip().lower()
        return out.startswith("y")

    # ── apply ─────────────────────────────────────────────────────────────────────
    def run(self) -> list[MergeDecision]:
        decisions = self.plan()
        for d in decisions:
            self.store.merge_entities(d.canonical, d.duplicate, record_history=True)
        return decisions


def _dedup_decisions(decisions: list[MergeDecision]) -> list[MergeDecision]:
    """Resolve transitive/duplicate merges so each duplicate is merged once."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out: list[MergeDecision] = []
    seen: set[str] = set()
    # apply alias seeds first (stable canonical targets)
    ordered = sorted(decisions, key=lambda d: 0 if d.reason == "alias_seed" else 1)
    for d in ordered:
        if d.duplicate in seen or d.duplicate == d.canonical:
            continue
        canon_root = find(d.canonical)
        parent[d.duplicate] = canon_root
        seen.add(d.duplicate)
        out.append(MergeDecision(canon_root, d.duplicate, d.string_score, d.embed_score, d.reason))
    return out
