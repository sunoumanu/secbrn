# Measuring Quality — Eval Harness

`secbrn eval` scores the brain against a **gold set** so schema/threshold/chunking
changes are judged by a number, not a vibe. It measures three layers; evaluate any
subset you have labels for.

```bash
secbrn eval --gold eval/gold.json              # isolated corpus, then score
secbrn eval --gold eval/gold.json --show-cases # per-query detail
secbrn eval --use-existing                      # score the live Neo4j store instead
secbrn eval --k 10                              # change the retrieval cutoff
```

## What is measured

### Retrieval
For each `query`, the gold set lists the **document titles** that should come back.

| metric | meaning | when to trust it |
|---|---|---|
| precision@k | fraction of the top-k that are relevant | misleading when relevant ≪ k |
| **precision@R** | precision at k = number of relevant docs | robust at any scale |
| recall@k | fraction of relevant docs found in top-k | coverage |
| **MAP** | mean average precision (rank-sensitive) | overall ranking quality |
| **nDCG@k** | discounted gain, rewards relevant-high | overall ranking quality |
| MRR | 1/rank of the first relevant hit | "is the first hit good" |
| hit@k | did any relevant doc appear | coarse sanity |

> **Read precision@R and nDCG@k first.** Plain precision@k is capped at ~`relevant/k`,
> so on a small corpus with 1 relevant doc and k=6 it can never exceed ~0.17 even with
> perfect ranking. That is a metric artifact, not a quality problem.

### Extraction
Each case gives `text` plus the gold `entities` `[name, label]` and `triples`
`[subject, relation, object]`. Reported as micro precision/recall/F1 over all cases.
This measures your **extraction model** — run with `SECBRN_PROVIDER=ollama` to score
the real LLM.

### Resolution
Each case lists `entities`, plus `should_merge` and `should_not_merge` pairs. Reported
as pairwise precision/recall/F1 where **FP = over-merge** (the dangerous failure) and
**FN = missed duplicate**.

## Isolation

By default retrieval is evaluated against a **throwaway in-memory copy** of just the
gold corpus, so unrelated documents already in your Neo4j can't dilute precision. Use
`--use-existing` to score the real store (realistic "needle in a haystack").

## Building a real gold set

Start small and grow it from real failures:

1. Collect 20–50 real questions you actually ask your brain.
2. For each, run `secbrn search "<q>"` and record the document titles that *should*
   be returned. That's your `relevant` list.
3. Label ~10 chunks' entities/triples for extraction, and a handful of known
   duplicate/non-duplicate entity pairs for resolution.
4. Re-run `secbrn eval` after any change to chunk size, schema, resolution thresholds,
   or `graph_boost`, and keep the gold set under version control so scores are
   comparable over time.

Gold-set schema lives in `secbrn/eval/dataset.py`. JSON and YAML are both accepted.

## Tuning knobs that move these numbers

- `SECBRN_CHUNK_SIZE` / `SECBRN_CHUNK_OVERLAP` — retrieval precision/recall.
- `SECBRN_RETRIEVE_TOP_K` / `SECBRN_RETRIEVE_HOPS` — recall vs. precision trade-off.
- `SECBRN_GRAPH_BOOST` — weight of graph-aware scoring (0 disables it). Boosts chunks
  whose entities sit within `hops` of the query's entities.
- `SECBRN_TITLE_BOOST` — weight for matching query terms against a chunk's document
  title / section heading (0 disables it).
- `SECBRN_RES_*` thresholds — resolution precision (over-merge) vs. recall (missed dup).
- `SECBRN_EXTRACT_MODEL` — a stronger local model raises extraction F1.
