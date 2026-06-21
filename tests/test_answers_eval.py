"""Answer-quality eval: loader, pure scorers, and end-to-end on the fake provider."""

from __future__ import annotations

import json

from secbrn.config import Settings
from secbrn.pipeline import Brain
from secbrn.graph.memory import InMemoryStore
from secbrn.providers.fake import FakeEmbedder, FakeLLM
from secbrn.eval.answers import (
    AnswerCase,
    AnswerEvaluator,
    key_fact_recall,
    lexical_f1,
    load_answer_set,
)


def _brain():
    s = Settings(provider="fake", graph_backend="memory")
    return Brain(settings=s, store=InMemoryStore(), embedder=FakeEmbedder(dim=64),
                 extract_llm=FakeLLM(), answer_llm=FakeLLM())


def test_key_fact_recall():
    ans = "Reranking reorders candidates and improves precision at k."
    assert key_fact_recall(ans, ["reranking improves precision"]) == 1.0
    assert key_fact_recall(ans, ["graph expansion over hops"]) == 0.0
    assert key_fact_recall(ans, []) == 1.0  # nothing to miss


def test_lexical_f1_bounds():
    assert lexical_f1("same words here", "same words here") == 1.0
    assert lexical_f1("", "") == 1.0
    v = lexical_f1("vector search retrieval", "retrieval reranking graph")
    assert 0.0 <= v <= 1.0


def test_load_answer_set(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"questions": [
        {"query": "q1", "expected": "e1", "key_facts": ["f1", "f2"]},
        {"query": "q2"},
    ]}), encoding="utf-8")
    cases = load_answer_set(p)
    assert len(cases) == 2
    assert cases[0].query == "q1" and cases[0].key_facts == ["f1", "f2"]
    assert cases[1].expected == ""


def test_answer_evaluator_end_to_end(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text(
        "# Retrieval\n\nReranking reorders the fused candidates and improves precision. "
        "Entity resolution merges duplicates like pgvector and PGVector.",
        encoding="utf-8",
    )
    b = _brain()
    b.ingest(doc)
    cases = [
        AnswerCase(query="What does reranking do?",
                   expected="Reranking reorders candidates and improves precision.",
                   key_facts=["reranking improves precision"]),
    ]
    report = AnswerEvaluator(b, judge=b.answer_llm, k=4).evaluate(cases)
    assert report.n == 1
    assert 0.0 <= report.judge_correct <= 5.0
    assert 0.0 <= report.judge_complete <= 5.0
    assert 0.0 <= report.key_fact_recall <= 1.0
    assert 0.0 <= report.lexical_f1 <= 1.0
    assert 0.0 <= report.grounded_rate <= 1.0
    assert len(report.per_case) == 1


def test_answer_evaluator_no_judge(tmp_path):
    doc = tmp_path / "n.md"
    doc.write_text("# T\n\nNeo4j relates to Ollama for local retrieval.", encoding="utf-8")
    b = _brain()
    b.ingest(doc)
    cases = [AnswerCase(query="How does Neo4j relate to Ollama?",
                        expected="Neo4j relates to Ollama for local retrieval.")]
    report = AnswerEvaluator(b, judge=None, k=4).evaluate(cases)
    assert report.judged_by_llm is False  # lexical fallback only
    assert report.n == 1
