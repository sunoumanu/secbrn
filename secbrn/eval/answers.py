"""Answer-quality evaluation (Stage 8) — score generated answers against gold answers.

The retrieval/extraction/resolution harness (``secbrn eval``) measures whether the
right *context* is found; this measures whether the final *answer* is any good. You
author a small set of questions with reference answers (and optional key facts), then
this runs ``brain.ask`` on each and grades the generated answer with several signals:

  - judge_correct / judge_complete (0-5): an LLM-as-judge compares the generated answer
    to your reference answer. Reference-based, so it rewards factual agreement, not just
    fluency. Falls back to lexical scores if no judge LLM / unparseable output.
  - key_fact_recall (0-1): fraction of your authored key facts that actually appear in
    the answer (lexical, fuzzy). The cheapest hard signal — fully offline.
  - lexical_f1 (0-1): token-overlap F1 between generated and reference answer.
  - grounded (bool): did the answer carry inline citations (and not trip the uncited
    guard)? An ungrounded answer can still be "correct" by luck — track it separately.

Gold file (JSON; YAML too if PyYAML is present)::

    {
      "questions": [
        {
          "query": "How does reranking relate to retrieval quality?",
          "expected": "Reranking reorders the fused candidates with an LLM, improving precision@k ...",
          "key_facts": ["reranking improves precision", "listwise LLM rerank"]
        }
      ]
    }

A bare top-level list of question objects is also accepted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from secbrn.eval import metrics as M
from secbrn.providers.base import LLM

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")
_STOP = {
    "the", "and", "for", "are", "was", "that", "this", "with", "from", "into", "have",
    "has", "had", "not", "but", "you", "your", "its", "their", "they", "them", "can",
    "will", "which", "what", "how", "does", "use", "used", "uses", "via", "per",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) >= 3 and t not in _STOP}


@dataclass
class AnswerCase:
    query: str
    expected: str = ""
    key_facts: list[str] = field(default_factory=list)
    top_k: int | None = None


@dataclass
class AnswerCaseResult:
    query: str
    generated: str
    judge_correct: float       # 0-5
    judge_complete: float      # 0-5
    key_fact_recall: float     # 0-1 (1.0 if no key facts given)
    lexical_f1: float          # 0-1
    grounded: bool
    n_citations: int
    judge_reason: str = ""


@dataclass
class AnswerEvalReport:
    n: int
    judge_correct: float
    judge_complete: float
    key_fact_recall: float
    lexical_f1: float
    grounded_rate: float
    judged_by_llm: bool
    per_case: list[AnswerCaseResult] = field(default_factory=list)


# ── loading ──────────────────────────────────────────────────────────────────────
def load_answer_set(path: str | Path) -> list[AnswerCase]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # optional
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    items = raw.get("questions", raw) if isinstance(raw, dict) else raw
    cases: list[AnswerCase] = []
    for c in items:
        cases.append(
            AnswerCase(
                query=c["query"],
                expected=str(c.get("expected", "")),
                key_facts=[str(f) for f in c.get("key_facts", [])],
                top_k=c.get("top_k"),
            )
        )
    return cases


# ── scoring helpers ────────────────────────────────────────────────────────────────
def key_fact_recall(answer: str, facts: list[str], threshold: float = 0.6) -> float:
    """Fraction of facts whose content words mostly appear in the answer.

    A fact counts as recalled when >= ``threshold`` of its content tokens are present
    in the answer (substring-tolerant), so phrasing differences don't tank the score.
    Returns 1.0 when no facts were authored (nothing to miss)."""
    if not facts:
        return 1.0
    atoks = _tokens(answer)
    a_lower = answer.lower()
    hits = 0
    for fact in facts:
        ftoks = _tokens(fact)
        if not ftoks:
            continue
        present = sum(1 for t in ftoks if t in atoks or t in a_lower)
        if present / len(ftoks) >= threshold:
            hits += 1
    return hits / len(facts)


def lexical_f1(generated: str, expected: str) -> float:
    """Token-set F1 between generated and reference answer. 1.0 if both empty."""
    g, e = _tokens(generated), _tokens(expected)
    if not g and not e:
        return 1.0
    return M.set_prf(g, e).f1


_JUDGE_SYSTEM = (
    "You are a strict grader of question-answering systems. Compare a CANDIDATE answer "
    "to a REFERENCE answer for the same question. Judge factual agreement with the "
    "reference, not style. Output STRICT JSON only."
)


def _judge_prompt(query: str, expected: str, generated: str) -> str:
    return (
        f"QUESTION:\n{query}\n\n"
        f"REFERENCE ANSWER:\n{expected}\n\n"
        f"CANDIDATE ANSWER:\n{generated}\n\n"
        "Score the CANDIDATE against the REFERENCE. Return JSON exactly like:\n"
        '{"correct": <0-5 integer, how factually consistent with the reference>, '
        '"complete": <0-5 integer, how much of the reference it covers>, '
        '"grounded": <true|false, does it read as supported rather than invented>, '
        '"reason": "<one short sentence>"}'
    )


def _safe_json(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip() if raw.count("```") >= 2 else raw
    a, b = raw.find("{"), raw.rfind("}")
    if a != -1 and b != -1:
        raw = raw[a:b + 1]
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _clamp5(x) -> float:
    try:
        return max(0.0, min(5.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ── evaluator ──────────────────────────────────────────────────────────────────────
class AnswerEvaluator:
    """Run + grade answers. ``judge`` is the LLM-as-judge (None => lexical only)."""

    def __init__(self, brain, judge: LLM | None = None, k: int | None = None):
        self.brain = brain
        self.judge = judge
        self.k = k

    def evaluate(self, cases: list[AnswerCase]) -> AnswerEvalReport:
        results: list[AnswerCaseResult] = []
        judged_any = False
        for case in cases:
            ans = self.brain.ask(case.query, top_k=case.top_k or self.k)
            gen = ans.text or ""
            kfr = key_fact_recall(gen, case.key_facts)
            lf1 = lexical_f1(gen, case.expected)
            grounded = (not ans.uncited) and len(ans.citations) > 0

            correct = complete = None
            reason = ""
            if self.judge is not None and case.expected:
                data = _safe_json(self.judge.complete_json(
                    _judge_prompt(case.query, case.expected, gen), system=_JUDGE_SYSTEM))
                if "correct" in data or "complete" in data:
                    judged_any = True
                    correct = _clamp5(data.get("correct", 0))
                    complete = _clamp5(data.get("complete", 0))
                    reason = str(data.get("reason", ""))[:200]

            # Fallback (no judge, no reference, or unparseable): derive 0-5 from the
            # lexical signals so the report is always populated and offline-safe.
            if correct is None:
                correct = round(5.0 * lf1, 2)
                complete = round(5.0 * (0.5 * lf1 + 0.5 * kfr), 2)
                reason = reason or "lexical fallback (no LLM judgment)"

            results.append(AnswerCaseResult(
                query=case.query, generated=gen,
                judge_correct=correct, judge_complete=complete,
                key_fact_recall=kfr, lexical_f1=lf1,
                grounded=grounded, n_citations=len(ans.citations), judge_reason=reason,
            ))

        return AnswerEvalReport(
            n=len(results),
            judge_correct=M.mean([r.judge_correct for r in results]),
            judge_complete=M.mean([r.judge_complete for r in results]),
            key_fact_recall=M.mean([r.key_fact_recall for r in results]),
            lexical_f1=M.mean([r.lexical_f1 for r in results]),
            grounded_rate=M.mean([1.0 if r.grounded else 0.0 for r in results]),
            judged_by_llm=judged_any,
            per_case=results,
        )
