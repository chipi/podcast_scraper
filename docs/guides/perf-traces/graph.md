# Graph perf traces

Capture harness for the gi-kg-viewer **Graph** route.

- **UI:** `scripts/dev/capture-graph-lcp.{sh,mjs}` — shell LCP, graph
  time-to-canvas, main-thread block.
- **API:** `scripts/dev/capture-graph-api.{sh,py}` *(Chunk 2 — pending)* —
  graph/relational endpoint latency, no browser.
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
| graph API latency | graph/relational endpoints, no browser | Chunk 2 |

## Trace archives

`*.trace.json.gz` (Chrome DevTools Performance format) are kept for run 1 of
each condition. Open in [perfetto](https://ui.perfetto.dev/) or
`chrome://tracing`. Raw `*.trace.json` (~70 MB) are gitignored; the gz is ~8 MB.

## Reports

The **graph-v3 tuning arc** (2026-07-19) — the first full graph perf report,
with the fcose/bridge/KG-second-wave analysis — is preserved at
[reports/graph-v3-tuning-2026-07-19.md](reports/graph-v3-tuning-2026-07-19.md).
Per-release numbers live under [reports/](reports/index.md).
