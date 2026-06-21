"""Tests for the eval harness + metrics."""

from __future__ import annotations

from pathlib import Path

from secbrn.eval import Evaluator, load_goldset
from secbrn.eval import metrics as M

FIX = Path(__file__).parent / "fixtures"
GOLD = Path(__file__).parent / "fixtures" / "gold_offline.json"


# ── metrics unit tests ──────────────────────────────────────────────────────────
def test_precision_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C"}
    assert M.precision_at_k(retrieved, relevant, 2) == 0.5      # A relevant, B not
    assert M.recall_at_k(retrieved, relevant, 4) == 1.0
    assert M.reciprocal_rank(retrieved, relevant) == 1.0       # A at rank 1
    assert M.reciprocal_rank(["B", "A"], relevant) == 0.5


def test_set_prf():
    prf = M.set_prf({("X", "Tool"), ("Y", "Concept")}, {("X", "Tool")})
    assert prf.tp == 1 and prf.fp == 1 and prf.fn == 0
    assert prf.precision == 0.5 and prf.recall == 1.0


def test_pairwise_resolution():
    clusters = {"pgvector": "pgvector", "PGVector": "pgvector", "Neo4j": "Neo4j"}
    prf = M.pairwise_resolution(clusters, [("pgvector", "PGVector")], [("pgvector", "Neo4j")])
    assert prf.tp == 1 and prf.fp == 0 and prf.fn == 0
    assert prf.f1 == 1.0


# ── harness integration tests (offline, scripted provider) ───────────────────────
def test_full_eval_over_fixtures(brain):
    brain.ingest(FIX)  # populate retrieval corpus
    gold = load_goldset(GOLD)
    report = Evaluator(brain).evaluate(gold)

    # retrieval: metrics well-formed, and at least one query finds a relevant doc
    assert report.retrieval is not None
    assert 0.0 <= report.retrieval.precision_at_k <= 1.0
    assert 0.0 <= report.retrieval.recall_at_k <= 1.0
    assert report.retrieval.hit_at_k > 0.0

    # extraction: the scripted LLM should nail these schema-clean cases
    assert report.triples.f1 == 1.0
    assert report.entities.f1 == 1.0

    # resolution: alias seeds + fuzzy should merge dupes without over-merging
    assert report.resolution.fp == 0          # no over-merges
    assert report.resolution.recall == 1.0    # caught all true duplicates


def test_eval_runs_with_only_resolution_section(brain):
    from secbrn.eval.dataset import GoldSet, ResolutionCase

    gs = GoldSet(resolution=[ResolutionCase(
        entities=[("pgvector", "Tool"), ("PGVector", "Tool"), ("Neo4j", "Tool")],
        should_merge=[("pgvector", "PGVector")],
        should_not_merge=[("pgvector", "Neo4j")],
    )])
    report = Evaluator(brain).evaluate(gs)
    assert report.retrieval is None and report.resolution is not None
    assert report.resolution.f1 == 1.0


# ── production gold set + A/B comparison ─────────────────────────────────────────
PROD_GOLD = Path(__file__).parent.parent / "eval" / "gold.json"


def _fake_isolated():
    from secbrn.config import Settings
    from secbrn.graph.memory import InMemoryStore
    from secbrn.pipeline import Brain
    from secbrn.providers.fake import FakeEmbedder, FakeLLM
    s = Settings(provider="fake", embed_dim=64, chunk_size=500, chunk_overlap=60)
    llm = FakeLLM()
    return Brain(settings=s, store=InMemoryStore(), embedder=FakeEmbedder(dim=64),
                 extract_llm=llm, answer_llm=llm)


def test_prod_gold_labels_reference_real_docs():
    """Every retrieval 'relevant' title must be an actual corpus document title."""
    gs = load_goldset(PROD_GOLD)
    brain = _fake_isolated()
    cp = gs.corpus_path()
    assert cp and cp.exists(), "corpus path missing"
    brain.ingest(cp, resolve=False)
    titles = {d.title for d in brain.store.documents.values()}
    gold_titles = {t for case in gs.retrieval for t in case.relevant}
    missing = gold_titles - titles
    assert not missing, f"gold references non-existent doc titles: {missing}"
    assert len(gs.retrieval) >= 25
    brain.close()


def test_ab_extract_model_override_runs():
    """A/B plumbing: two extract-model settings copies both evaluate end-to-end."""
    from secbrn.config import get_settings
    from secbrn.pipeline import Brain

    gs = load_goldset(PROD_GOLD)
    base = get_settings().model_copy(update={"provider": "fake", "embed_dim": 64})
    reports = []
    for m in ("model-a", "model-b"):
        s2 = base.model_copy(update={"extract_model": m})
        assert s2.extract_model == m
        brain = Brain.isolated(s2)
        try:
            brain.ingest(gs.corpus_path(), resolve=False)
            reports.append(Evaluator(brain).evaluate(gs))
        finally:
            brain.close()
    assert all(r.retrieval is not None for r in reports)
    assert all(r.entities is not None for r in reports)
