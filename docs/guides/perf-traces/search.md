# Search perf traces

Capture harness for the gi-kg-viewer **Search** surface (Query Workspace,
RFC-107). Two capture paths: **UI** (browser paint/interaction timings) and
**API** (endpoint latency, no browser).

- **UI:** `scripts/dev/capture-search-perf.{sh,mjs}`
- **API:** `scripts/dev/capture-search-api.{sh,py}`
- **Deep recipe:** [SEARCH_PERF_TRACE_RUNBOOK.md](../SEARCH_PERF_TRACE_RUNBOOK.md)
- **Raw artifacts:** `data/perf/traces/search/`

See [index.md](index.md) for the framework, standard conditions (prod-v2,
1440×900 @ DPR-2, median-of-3), and the split rationale.

## Run it

```bash
# UI scenarios (boots dedicated api + viewer, tears down):
scripts/dev/capture-search-perf.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label <release>-search-ui \
  --output-dir data/perf/traces/search

# API-only latency:
scripts/dev/capture-search-api.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --queries tests/fixtures/viewer-validation-corpus/v3/search-queries.json \
  --label <release>-search-api \
  --output-dir data/perf/traces/search
```

## Scenario catalog (UI)

Status reflects what `capture-search-perf.mjs` implements today. RFC-107 §P2
enumerated the full target set; Chunk 1 of the perf-resolidify pass fills the
S2–S8 scenarios.

| Scenario | Measures | Status |
| --- | --- | --- |
| `workspace-open` (TTI) | page load → set corpus path → Search tab → `#search-q` visible | Implemented |
| `filter-apply` | top-k chip click → popover visible | Implemented |
| `results-paint` | submit query → first result card visible | Implemented |
| `cmdk-open` | `/` → palette overlay visible (S3) | Implemented |
| `operator-cluster` | Cluster chip → cluster panel populated (S4) | Implemented |
| `operator-compare` | Compare run → 2-column packs rendered (S8) | Implemented |

`--query <q>` overrides the results/operator query (default `the economy`, suited
to the prod-v2 finance corpus). Every scenario degrades gracefully (records an
`error` field, never aborts the run) so a report is always produced. Latest run:
`data/perf/traces/search/2.7.0.dev1-search-ui.ui.metrics.json`.

## Scenario catalog (API)

`capture-search-api.py` emits one aggregated `*.api.metrics.json` of per-query
latency over the labelled query set (see the runbook). The committed
`S4-shell-fixture-baseline.api.metrics.json` is the S4-shell baseline.

## Reports

Per-release numbers + comparison tables live under
[reports/](reports/index.md).
