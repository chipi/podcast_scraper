# search-queries.json — Search v3 labelled query set

**Owner:** Search v3 arc (epic #1229 · plan `docs/wip/SEARCH-V3-IMPLEMENTATION-PLAN.md` §S0)
**Consumed by:** `scripts/eval/search_quality.py` (driven by `make eval-search`)
**Target corpus:** this directory (`tests/fixtures/viewer-validation-corpus/v3/`)

## What it is

25 queries covering all 5 RFC-092 intent classes (5 per class:
`entity_lookup`, `raw_evidence`, `temporal_tracking`, `cross_show_synthesis`,
`semantic`). Each entry has:

- `id` — stable identifier (e.g. `el_003` = entity_lookup #3).
- `q` — the query text.
- `intent_expected` — my honest classification per RFC-092. The router's actual
  classification may differ; the delta is the intent-router-accuracy metric.
- `expected_top_k_doc_ids` — the labelled top-K (or `null` if unlabeled).
- `min_ndcg_at_10` — per-query nDCG@10 floor (RFC-107 §T2 requires per-query
  floors for regression protection).
- `label_status` — one of:
  - `unlabeled-seed` — `expected_top_k_doc_ids` is `null`; nDCG@10 skipped for this query.
  - `regression-anchor` — `expected_top_k_doc_ids` was frozen from what the
    harness returned on a specific `search-v3` commit; drift detection only.
  - `human-audit` — `expected_top_k_doc_ids` was hand-picked by an auditor
    against the corpus content; correctness detection.
- `hint` — optional; a short note about why the query is here / what it targets
  in the fixture.

## Current state (2026-07-20, S0 baseline ship)

**Every query is `unlabeled-seed` — `expected_top_k_doc_ids` is `null`.**

Nobody has hand-audited the fixture's index yet. Shipping the query text +
intent classifications separately from the labels is the honest split: the
router-accuracy metric + tier-coverage + compound-lift-rate + enriched-answer
groundedness rate all still run (they don't need labels); nDCG@10 and MRR@10
skip queries where labels are missing (per-query, reported in the summary).

RFC-107 §T2 asks for a **labelled baseline** — that lands the moment either:

1. Someone (Marko, another operator, or Claude in a follow-up session)
   runs `make eval-search seed-labels` (mode TBD) to REGRESSION-ANCHOR
   every query's `expected_top_k_doc_ids` to the current top-K returned by
   the harness on the fixture. Detects drift from that day's ranking.
2. A human hand-audits each query against the fixture's insights + topics +
   people and picks the correct answers (`label_status: human-audit`).
   Detects correctness.

Both are useful; (1) is fastest to produce.

## Why the S0 ship intentionally does NOT seed the labels

RFC-107 §T2 explicitly says the baseline is a REGRESSION protector. But
regression-anchor labels frozen the day they're generated can hide bugs
present that day — they treat the harness's current output as ground truth.
Marko has been explicit (T3, T7 in the truthfulness protocol) about not
inventing correctness where none was validated. So the S0 ship carries the
scaffolding for both label lifecycles and leaves the seed step for a
subsequent operator-owned pass.

The metrics that don't depend on labels still baseline today:
- intent-router accuracy (measurable — we predict vs. `intent_expected`).
- tier coverage (measurable — count of insight/segment/aux per response).
- compound-lift rate (measurable — count of `lifted` blocks on transcript
  hits).
- enriched-answer groundedness (measurable when a provider is configured).
- `topic_consensus` precision (measurable when consensus enricher output
  exists on the fixture — currently doesn't, so this metric is null).

## Lifecycle

```
scaffold (this ship)  →  regression-anchor (seed labels)  →  human-audit
     ↓                          ↓                                ↓
label_status:          label_status:                    label_status:
"unlabeled-seed"       "regression-anchor"              "human-audit"
     ↓                          ↓                                ↓
nDCG@10 skipped        nDCG@10 detects DRIFT            nDCG@10 detects
                       (from the anchor commit)         CORRECTNESS
```

## Adding / editing queries

1. Preserve `id` stability — external eval reports may reference `id`.
2. Add new queries at the end of the array in the intent-class group.
3. If you re-classify `intent_expected`, note the reason in `hint`.
4. Never delete a query — it breaks historical comparisons. Mark
   `label_status: "retired"` if a query is no longer relevant (the harness
   will skip it but the record survives).

## Related

- `scripts/eval/search_quality.py` — the harness that consumes this file.
- `make eval-search` — the CI target.
- RFC-107 §T2 — the metric spec.
- RFC-092 — the intent taxonomy.
- ADR-108 — `topic_consensus` context for the precision metric.
