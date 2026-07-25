# Graph perf traces

Capture harness for the gi-kg-viewer **Graph** route.

- **UI:** `scripts/dev/capture-graph-lcp.{sh,mjs}` — shell LCP, graph
  time-to-canvas, main-thread block.
- **API:** `scripts/dev/capture-graph-api.{sh,py}` — graph-load endpoint
  latency (artifacts list + per-episode GI/KG fan-out + topic-clusters +
  4-way concurrency guard), no browser. Targets a running api (like
  capture-search-api).
- **Deep recipe:** [GRAPH_PERF_TRACE_RUNBOOK.md](../GRAPH_PERF_TRACE_RUNBOOK.md)
- **Raw artifacts:** `data/perf/traces/graph/`

See [index.md](index.md) for the framework, standard conditions, and split
rationale.

## Run it

```bash
# UI (LCP / time-to-canvas), default 'everything' load mode:
scripts/dev/capture-graph-lcp.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label <release>-graph-ui \
  --output-dir data/perf/traces/graph \
  --wait-ms 6000

# topDown load mode + expand-on-tap probe (dev build for window.__GIKG_CY_DEV__):
scripts/dev/capture-graph-lcp.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label <release>-graph-topdown \
  --load-mode topDown --expand-first-super-theme --wait-ms 8000
```

## Scenario catalog

| Scenario | Measures | Status |
| --- | --- | --- |
| shell LCP | Web-Vitals LCP of the app shell | Implemented |
| `graph_time_to_canvas_ms` | Graph tab click → `.graph-canvas` has cy children | Implemented |
| topDown mount | 6-SuperTheme rollup time-to-canvas | Implemented |
| expand-on-tap | SuperTheme tap → resettle (fast-path) | Implemented |
| `api-artifacts-list` | GET /api/artifacts (corpus envelope listing) | Implemented |
| `api-artifact-fetch` | GET /api/artifacts/&lt;relpath&gt; fan-out (GI/KG) | Implemented |
| `api-topic-clusters` | GET /api/corpus/topic-clusters | Implemented |
| `api-concurrent-4` | 4-way artifact fan-out (SIGSEGV/stability guard) | Implemented |

**API-only run:**

```bash
# api must be running (make serve / podcast-scraper-api --path <corpus>):
scripts/dev/capture-graph-api.sh \
  --api http://127.0.0.1:8000 \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label <release>-graph-api \
  --output-dir data/perf/traces/graph
```

Latest run `data/perf/traces/graph/2.7.0.dev1-graph-api.api.metrics.json`: server
artifact fetch is ~0 ms (fast file serves) — confirming the graph-v3 report's
finding that graph time-to-canvas is dominated by **client-side** parse/merge/fcose,
not server latency. The artifacts-list is the notable server cost (~108 ms p50).

## Trace archives

`*.trace.json.gz` (Chrome DevTools Performance format) are kept for run 1 of
each condition. Open in [perfetto](https://ui.perfetto.dev/) or
`chrome://tracing`. Raw `*.trace.json` (~70 MB) are gitignored; the gz is ~8 MB.

## Reports

The **graph-v3 tuning arc** (2026-07-19) — the first full graph perf report,
with the fcose/bridge/KG-second-wave analysis — is preserved at
[reports/graph-v3-tuning-2026-07-19.md](reports/graph-v3-tuning-2026-07-19.md).
Per-release numbers live under [reports/](reports/index.md).
