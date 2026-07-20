# Search v3 performance trace runbook

**Purpose:** capture reproducible perf measurements for the Search v3 arc
(epic #1229 · RFC-107 §P) — both the API surface (this document, ships in
S0) and the UI Query Workspace (deferred to slice S2 when the Workspace
actually exists). Mirrors the sibling
[GRAPH_PERF_TRACE_RUNBOOK.md](GRAPH_PERF_TRACE_RUNBOOK.md) shape so
operators only need to learn one recipe.

**When to use:**

- Before + after any change under `src/podcast_scraper/search/` — retrieval,
  fusion, backend, router, enrichers touched by `/api/search`.
- Before + after enabling / disabling an enricher that runs on the
  `/api/search?enrich_results=true` path (RFC-088 chunk 5).
- Periodically against `main` to catch drift.
- Every slice PR under #1229 that touches any of the above (per RFC-107 §P3).

**When NOT to use:**

- Unit / integration perf — those use pytest timers.
- Frontend rendering perf — that's `capture-search-perf.{sh,mjs}`, which
  lands with slice S2 (see [Deferred](#deferred-ui-side-capture) below).
- Non-Search v3 API perf — has no baselining rig; add one if it becomes
  a recurring need.

---

## 1. Prerequisites

- `.venv/bin/python` present (`make dev-setup`).
- A running api (`make serve` locally, or any URL reachable from your host).
- Corpus at a known path with a LanceDB index the api can serve.
- The labelled query set at
  `tests/fixtures/viewer-validation-corpus/v3/search-queries.json`
  (ships in S0 alongside this runbook).

Recommended combo for a stable baseline: `make serve` against
`.test_outputs/manual/prod-v2/corpus` (833-node reference from the graph
runbook) or against the checked-in synthetic corpus at
`tests/fixtures/viewer-validation-corpus/v3/`.

## 2. The scripts

### 2.1 `scripts/dev/capture-search-api.sh` (ships in S0)

Pure-HTTP perf capturer for `/api/search`. Runs the labelled query set
against a running api and records latency + concurrent-load behavior.

- **Orchestrator (`capture-search-api.sh`):** validates args, checks api
  reachability at `/api/health`, hands off to the Python capturer.
- **Capturer (`capture_search_api.py`):** runs 12 scenarios by default:
  - 5 `api-intent-<class>` — per-intent latency (top_k=10) across all 5
    RFC-092 intent classes.
  - 4 `api-top_k-{10,25,50,100}` — latency vs top_k on a stable
    3-query subset.
  - 1 `api-concurrent-4` — 4-worker × 5-query parallel load. Doubles as
    the runtime companion to `make lint-search-v3` — a socket-level
    failure here is the HTTP-side signature of the #1205 SIGSEGV (killed
    api worker) and fails the run non-zero.

Emits one aggregated `<label>.api.metrics.json` under `--output-dir`
(default `docs/wip/search-v3/traces/`) with p50/p95/p99/max/mean per
scenario + a `sigsegv_free` flag on the concurrent scenario.

### 2.2 `scripts/dev/capture-search-perf.{sh,mjs}` (deferred to slice S2)

The UI-side CDP capturer. Follows the graph-v3 pattern exactly (isolated
api + viewer subprocesses, Playwright + CDP trace, median-of-3), but is
not authored here because the target UX — the Query Workspace, Cmd-K
palette, operator bar — doesn't exist yet. Building it now would either:

1. Trace the current LeftPanel Search — real but not what we care about
   (S1/S2 make it obsolete anyway).
2. Print NOT_APPLICABLE_YET stubs — worse than nothing.

**Ships with slice S2 (#1232, Query Workspace shell).** By then there is
something meaningful to trace; the baseline captured against slice-S2 tip
becomes the "before" for slices S3–S8.

## 3. Common recipes

### Baseline capture (ship at end of S0)

```bash
# 1. Boot a dedicated api against your corpus.
make serve  # or: podcast-scraper-api --path /abs/path/to/corpus

# 2. Capture the S0 API baseline.
scripts/dev/capture-search-api.sh \
    --api http://localhost:8000 \
    --corpus /abs/path/to/corpus \
    --queries tests/fixtures/viewer-validation-corpus/v3/search-queries.json \
    --label S0-api-baseline \
    --iterations 3

# 3. Commit the JSON.
git add docs/wip/search-v3/traces/S0-api-baseline.api.metrics.json
```

### Per-slice diff (every slice PR under #1229 that touches search)

```bash
scripts/dev/capture-search-api.sh \
    --api http://localhost:8000 \
    --corpus <same corpus as baseline> \
    --queries tests/fixtures/viewer-validation-corpus/v3/search-queries.json \
    --label S<N>-<slice-slug>-tip \
    --iterations 3

# Diff against the S0 baseline in the PR body:
diff <(jq -S . docs/wip/search-v3/traces/S0-api-baseline.api.metrics.json) \
     <(jq -S . docs/wip/search-v3/traces/S<N>-<slice-slug>-tip.api.metrics.json)
```

## 4. Reading the outputs

Each run writes one JSON to `<output-dir>`:

```jsonc
{
  "schema_version": "1",
  "label": "S0-api-baseline",
  "captured_at": "2026-07-20T…Z",
  "api": "http://localhost:8000",
  "corpus": "/abs/path/…",
  "queries_path": "tests/fixtures/…/search-queries.json",
  "iterations": 3,
  "scenarios": [
    {
      "name": "api-intent-entity_lookup",
      "iterations": 3,
      "request_count": 15,
      "ok_count": 15,
      "p50_ms": 42, "p95_ms": 68, "p99_ms": 72, "max_ms": 72, "mean_ms": 44.3,
      "sigsegv_free": null
    },
    // …
    {
      "name": "api-concurrent-4",
      "request_count": 60,
      "sigsegv_free": true,
      // …
    }
  ]
}
```

Only the aggregated file is checked in per capture. There is no raw trace
sidecar (the graph runbook has `.trace.json.gz` because Chrome DevTools
traces are big; HTTP latency arrays are cheap and live inline).

## 5. Fair-comparison invariants

- **Same corpus.** Index shape dominates BM25/vector latency.
- **Same api process.** Cold vs warm matters — run a warm-up query
  before the measured iterations (the capturer does one implicit warmup
  when it hits `/api/health`; add explicit warmup queries if you observe
  first-iteration outliers).
- **Same query set.** The labelled query set at
  `tests/fixtures/viewer-validation-corpus/v3/search-queries.json` is the
  stable input; if you change it, note the diff in the PR body.
- **Same machine state.** Close heavy workloads before capture.
- **Median-of-N.** Default `--iterations 3`; report all three, note the
  median in the PR body.

## 6. What "counts" for the baseline

For any perf claim on a Search v3 slice PR:

- **The metric is p95_ms per intent** (regression protector — RFC-107
  §Success metrics: "API p95 per intent ≤ baseline").
- **The concurrent-4 scenario must have `sigsegv_free: true`** — a
  single false there means #1205 regressed; block the merge.
- **The corpus is the operator's real corpus** unless the change
  specifically targets fixture-corpus behavior.
- **Report the delta with sign per intent.** "entity_lookup p95:
  42ms → 55ms (+31%, regression)".
- **Include the `.api.metrics.json` in the commit** so the number can
  be audited without re-running.

## 7. Troubleshooting

**"api port 8000 already in use"** — kill the holder or point `--api`
at a different URL. The graph runbook uses isolated ports :8600/:5600
via a helper script; the search API capturer trusts you to point it at
a reachable api rather than boot its own (simpler, matches how you'd run
it against a staging api).

**"api unreachable at .../api/health"** — the health check failed. Run
`curl -v <api>/api/health` to diagnose (503 = no corpus configured;
connection refused = api not up).

**p95 wildly different from expected** — is the LanceDB FTS index built?
`curl <api>/api/index/stats | jq` should show `insight`/`segment`/`aux`
tables with `rows > 0` and `has_fts_index: true` on each. If any is
absent, run the reindex CLI or start the api with `--rebuild-index`.

**`api-concurrent-4` shows `sigsegv_free: false`** — this is #1205
regressing at runtime. Immediately: (1) revert the offending change,
(2) re-run `make lint-search-v3` (should catch a compile-time re-import
of `_combine_hybrid_results` / `_normalize_scores`), (3) if lint is
green, the regression is subtler than an import — file an incident
before re-attempting the change.

## 8. Related

- [GRAPH_PERF_TRACE_RUNBOOK.md](GRAPH_PERF_TRACE_RUNBOOK.md) — the
  template this runbook mirrors.
- `scripts/eval/search_quality.py` — the quality-eval harness (S0(c));
  runs against the SAME labelled query set as `capture-search-api.sh`.
- `tests/integration/search/test_lancedb_concurrent_no_native_combine.py` —
  the compile-time-plus-in-process SIGSEGV guardrail (S0(b));
  complements the `api-concurrent-4` HTTP-side assertion here.
- `.github/lint/search-v3-forbidden-imports.txt` — the forbidden-imports
  rule the whole guardrail stack enforces (S0(a)).
- RFC-107 §P — the perf-capture spec this runbook implements.
- PRD-045 FR11 — the requirement source.

## Deferred: UI-side capture

`scripts/dev/capture-search-perf.{sh,mjs}` — CDP-based Playwright
capturer for the 6 UI scenarios listed in RFC-107 §P2 (Workspace-open
TTI, Cmd-K open latency, filter-apply, operator-cluster,
operator-graph, enriched-answer-paint) — lands with slice S2
(#1232 · Query Workspace shell). By then there is a Workspace to
trace; the baseline captured against slice-S2 tip becomes the "before"
for slices S3–S8.

## Deferred: RFC-107 §P3 deep-review pass

Once Marko (or another operator) captures the S0-api-baseline against
a real corpus (recipe under §3 above), the RFC-107 §P3 "deep-review
pass" fires — a per-surface review of the top-N latency contributors
observed in the baseline, with action items opened per slice as they
apply. Convention (RFC-107 §P3): "Do NOT silently apply optimizations
discovered in the review." When the baseline lands, add the review
findings to this runbook under a new "§P3 deep-review findings"
section rather than opening a new tracking issue.
