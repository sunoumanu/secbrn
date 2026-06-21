"""Evaluation metrics — pure functions, no I/O, easy to unit-test.

Retrieval:  precision@k, recall@k, reciprocal rank (→ MRR when averaged).
Set tasks (extraction): precision / recall / F1 over predicted vs. gold sets.
Resolution: pairwise precision / recall / F1 from a merge clustering.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    fn: int = 0


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF(p, r, f1, tp, fp, fn)


def set_prf(predicted: set, gold: set) -> PRF:
    """Precision/recall/F1 treating items as a set (order-independent)."""
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return prf_from_counts(tp, fp, fn)


# ── retrieval ──────────────────────────────────────────────────────────────────
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r in relevant)
    return hits / len(topk)


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list, relevant: set, k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def r_precision(retrieved: list, relevant: set) -> float:
    """Precision at k=R where R=|relevant|. Undistorted when relevant << fixed k."""
    r = len(relevant)
    if r == 0:
        return 0.0
    return precision_at_k(retrieved, relevant, r)


def average_precision(retrieved: list, relevant: set) -> float:
    """AP: mean of precision@i taken at each rank i where a relevant item is hit."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def dcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    import math

    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)  # binary gain
    return dcg


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    import math

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── resolution (pairwise) ──────────────────────────────────────────────────────
def pairwise_resolution(
    clusters: dict[str, str],
    should_merge: list[tuple[str, str]],
    should_not_merge: list[tuple[str, str]],
) -> PRF:
    """Score a resolution outcome.

    ``clusters`` maps each entity surface form → its canonical name after resolution.
    A pair is "merged" iff both map to the same canonical. Positives are
    ``should_merge`` pairs; negatives are ``should_not_merge`` pairs.
    """

    def merged(a: str, b: str) -> bool:
        return clusters.get(a, a) == clusters.get(b, b)

    tp = sum(1 for a, b in should_merge if merged(a, b))
    fn = sum(1 for a, b in should_merge if not merged(a, b))
    fp = sum(1 for a, b in should_not_merge if merged(a, b))
    return prf_from_counts(tp, fp, fn)
