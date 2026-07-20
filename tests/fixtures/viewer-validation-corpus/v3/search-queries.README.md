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

## Current state (2026-07-20, S0 stabilization ship)

**Every query is `regression-anchor` — `expected_top_k_doc_ids` filled from
the harness's top-K on the seed-run commit.** Detects drift; NOT correctness.

Seeding was produced by `--seed-labels` (see below); the run that produced
the anchors trivially scored `nDCG@10 = 1.000` by construction. Any subsequent
`make eval-search` run whose top-K differs from these anchors will report a
lower nDCG@10 — that is the regression signal.

**A human audit still hasn't happened.** Regression-anchor labels can hide
bugs present the day they were frozen. When that audit lands, flip queries
from `regression-anchor` to `human-audit` (or delete + re-add) — the harness
respects both.

## Seeding labels (`--seed-labels` mode)

`scripts/eval/search_quality.py --seed-labels` freezes the current top-K
of every `unlabeled-seed` query into that query's
`expected_top_k_doc_ids` and flips `label_status` to `regression-anchor`.

- Idempotent: queries with any other `label_status` are left alone. Re-running
  is a no-op.
- Atomic: writes to `<queries>.tmp` then renames, so a crash mid-write
  cannot leave the file half-updated.
- Truncated to `--top-k` (default 10): the fused list may exceed it after
  dedup; the anchor is only the top-K worth.
- Ships alongside a per-run report (`--out`) that shows the labeled metrics
  — the seed run always sees `nDCG@10 = 1.000`, correctly.

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
