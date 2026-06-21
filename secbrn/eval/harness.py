"""Eval harness — runs a GoldSet against a Brain and reports per-layer metrics.

Layers (any subset, depending on what the gold set contains):
  - retrieval  : precision@k, recall@k, MRR, hit@k over query -> relevant-doc labels
  - extraction : micro precision/recall/F1 of entities and triples vs. gold
  - resolution : pairwise precision/recall/F1 of should-merge / should-not-merge pairs

Everything is deterministic given a fixed corpus + models, so re-running after a
schema/threshold/chunking change gives you a number to defend the change against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from secbrn.eval import metrics as M
from secbrn.eval.dataset import GoldSet
from secbrn.extract import extract_chunk
from secbrn.graph.memory import InMemoryStore
from secbrn.resolve import EntityResolver


@dataclass
class RetrievalResult:
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_at_k: float
    n: int
    r_precision: float = 0.0
    map: float = 0.0
    ndcg_at_k: float = 0.0
    per_case: list[dict] = field(default_factory=list)


@dataclass
class EvalReport:
    retrieval: RetrievalResult | None = None
    entities: M.PRF | None = None
    triples: M.PRF | None = None
    resolution: M.PRF | None = None


class Evaluator:
    def __init__(self, brain, k: int | None = None):
        self.brain = brain
        self.k = k or brain.s.retrieve_top_k

    def evaluate(self, gold: GoldSet) -> EvalReport:
        report = EvalReport()
        if gold.retrieval:
            report.retrieval = self._eval_retrieval(gold)
        if gold.extraction:
            report.entities, report.triples = self._eval_extraction(gold)
        if gold.resolution:
            report.resolution = self._eval_resolution(gold)
        return report

    # ── retrieval ────────────────────────────────────────────────────────────────
    def _eval_retrieval(self, gold: GoldSet) -> RetrievalResult:
        ps, rs, rr, hits, rps, aps, ndcgs, per_case = [], [], [], [], [], [], [], []
        for case in gold.retrieval:
            chunks = self.brain.search(case.query, top_k=self.k)
            # gold relevance is expressed by document title (see docs/EVAL.md). We also
            # accept a uri match, normalizing it to the doc title for scoring.
            relevant_in = set(case.relevant)
            ranked: list[str] = []
            seen: set[str] = set()
            title_of_uri = {}
            for rc in chunks:
                title_of_uri[rc.uri] = rc.document_title
                if rc.document_title not in seen:
                    seen.add(rc.document_title)
                    ranked.append(rc.document_title)
            # map any uri-form gold labels to their retrieved title
            relevant = {title_of_uri.get(x, x) for x in relevant_in}

            p = M.precision_at_k(ranked, relevant, self.k)
            r = M.recall_at_k(ranked, relevant, self.k)
            mrr = M.reciprocal_rank(ranked, relevant)
            h = M.hit_at_k(ranked, relevant, self.k)
            rp = M.r_precision(ranked, relevant)
            ap = M.average_precision(ranked, relevant)
            ndcg = M.ndcg_at_k(ranked, relevant, self.k)
            ps.append(p); rs.append(r); rr.append(mrr); hits.append(h)
            rps.append(rp); aps.append(ap); ndcgs.append(ndcg)
            per_case.append({
                "query": case.query, "precision": p, "recall": r, "rr": mrr,
                "r_precision": rp, "ap": ap, "ndcg": ndcg,
                "retrieved": ranked[: self.k], "relevant": list(relevant_in),
            })
        return RetrievalResult(
            k=self.k,
            precision_at_k=M.mean(ps),
            recall_at_k=M.mean(rs),
            mrr=M.mean(rr),
            hit_at_k=M.mean(hits),
            r_precision=M.mean(rps),
            map=M.mean(aps),
            ndcg_at_k=M.mean(ndcgs),
            n=len(gold.retrieval),
            per_case=per_case,
        )

    # ── extraction ───────────────────────────────────────────────────────────────
    def _eval_extraction(self, gold: GoldSet):
        e_tp = e_fp = e_fn = 0
        t_tp = t_fp = t_fn = 0
        for case in gold.extraction:
            ex = extract_chunk(case.text, self.brain.extract_llm)
            pred_ents = {(e.name, e.label) for e in ex.entities}
            gold_ents = {(n, l) for n, l in case.entities}
            ep = M.set_prf(pred_ents, gold_ents)
            e_tp += ep.tp; e_fp += ep.fp; e_fn += ep.fn

            pred_tr = {(r.subject, r.relation.upper(), r.object) for r in ex.relations}
            gold_tr = {(s, rel.upper(), o) for s, rel, o in case.triples}
            tp = M.set_prf(pred_tr, gold_tr)
            t_tp += tp.tp; t_fp += tp.fp; t_fn += tp.fn
        return M.prf_from_counts(e_tp, e_fp, e_fn), M.prf_from_counts(t_tp, t_fp, t_fn)

    # ── resolution ───────────────────────────────────────────────────────────────
    def _eval_resolution(self, gold: GoldSet) -> M.PRF:
        tp = fp = fn = 0
        for case in gold.resolution:
            store = InMemoryStore()
            for name, label in case.entities:
                store.upsert_entity(name, label, [])
                store.add_mention(f"c::{name}", name)  # give it a mention so it's "real"
            EntityResolver(self.brain.s, store).run()
            clusters = self._clusters(store, [n for n, _ in case.entities])
            r = M.pairwise_resolution(clusters, case.should_merge, case.should_not_merge)
            tp += r.tp; fp += r.fp; fn += r.fn
        return M.prf_from_counts(tp, fp, fn)

    @staticmethod
    def _clusters(store: InMemoryStore, names: list[str]) -> dict[str, str]:
        """Map every original surface form -> its canonical name after resolution."""
        parent = {n: n for n in names}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for dup, canon in store.same_as:  # recorded merges (duplicate -> canonical)
            parent.setdefault(dup, dup)
            parent.setdefault(canon, canon)
            parent[find(dup)] = find(canon)
        return {n: find(n) for n in names}
