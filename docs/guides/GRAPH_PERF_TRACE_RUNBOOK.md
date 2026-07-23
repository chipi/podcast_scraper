# Graph performance trace runbook

**Purpose:** capture reproducible Chrome DevTools Performance recordings of
the gi-kg-viewer graph route so we can compare LCP / FCP / TTI / main-thread
blocking time across git refs, layout parameter changes, or corpora. Every
trace lands in `docs/wip/graph-v3/traces/` under a labelled name so the
history stays browsable.

**When to use:**

- Before + after a fcose / layout parameter change (defend the tuning).
- Before + after any change that touches `GraphCanvas.vue`, `cyGraphStylesheet.ts`,
  or the merged-artifact assembly pipeline.
- Periodically against `main` to catch regressions early.
- **Not** for A/B on a hunch — capture only when you have a specific
  hypothesis or a required defense.

**When NOT to use:**

- Unit or integration test perf — those use vitest / pytest timers, different
  concern.
- Backend perf — this only measures viewer paint.

---

## 1. Prerequisites

- `.venv/bin/python` present (run `make init` if missing).
- `web/gi-kg-viewer/node_modules` present. Missing → the script auto-installs
  when `--ref` is given; on the current worktree, run `cd web/gi-kg-viewer && env -u NODE_OPTIONS npm ci`.
- Corpus with real data at a known path. For prod-comparable numbers use
  `.test_outputs/manual/prod-v2/corpus` (833 nodes on Tier-C baseline). The
  v3 synthetic fixture at `tests/fixtures/viewer-validation-corpus/v3` works
  but is too small to expose perf issues that only fire on real corpora.
- Chromium via Playwright — installed with `@playwright/test`; no extra
  setup.

## 2. The two scripts

- **`scripts/dev/capture-graph-lcp.sh`** — orchestrator. Boots isolated api
  (`:8600` default) and viewer (`:5600` default) servers, waits for both to
  be healthy, runs the mjs capturer, tears everything down. Ports are
  chosen to be unusual so it never collides with the operator's dev server.
  If a port IS taken it fails loudly with the process holding it.
- **`scripts/dev/capture-graph-lcp.mjs`** — the actual capture. Launches
  headless Chromium via Playwright, opens a CDP session, calls
  `Tracing.start` with the Performance panel default category set, navigates
  the graph route, waits for LCP + optional extra settle time, stops the
  trace, dumps chunks to `${label}.trace.json`, gzips to
  `${label}.trace.json.gz`, writes summary metrics to `${label}.metrics.json`.

## 3. Common recipes

### Baseline against `main`

```bash
scripts/dev/capture-graph-lcp.sh \
  --corpus $(pwd)/.test_outputs/manual/prod-v2/corpus \
  --ref main \
  --label main-baseline
```

Materialises `main` in a temporary git worktree so the current uncommitted
state is untouched. Worktree is cleaned up on exit.

### Current worktree (branch tip)

```bash
scripts/dev/capture-graph-lcp.sh \
  --corpus $(pwd)/.test_outputs/manual/prod-v2/corpus \
  --label $(git rev-parse --abbrev-ref HEAD)-tip
```

Omit `--ref` and the script traces the current worktree — useful when you're
mid-branch and want to check drift against the baseline you captured earlier.

### fcose parameter sweep

```bash
# baseline current
scripts/dev/capture-graph-lcp.sh --corpus $(pwd)/.test_outputs/manual/prod-v2/corpus --label fcose-A-baseline

# edit web/gi-kg-viewer/src/utils/cyCoseLayoutOptions.ts
# capture candidate
scripts/dev/capture-graph-lcp.sh --corpus $(pwd)/.test_outputs/manual/prod-v2/corpus --label fcose-B-repulsion-660k

# repeat, or diff the metrics JSONs
diff <(jq . docs/wip/graph-v3/traces/fcose-A-baseline.metrics.json) \
     <(jq . docs/wip/graph-v3/traces/fcose-B-repulsion-660k.metrics.json)
```

### CI trending later (out of scope now)

The script is intentionally CI-safe: no interactive prompts, hard-fails on
port collision, cleans up on any exit. A future GH Actions job can call it
against a checked-in synthetic corpus and post a comment on regression.

## 4. Reading the outputs

Each run drops **three** files into `--output-dir` (default
`docs/wip/graph-v3/traces/`):

- **`${label}.trace.json`** — raw Chrome DevTools trace (`traceEvents` shape).
  Open at [chrome://tracing](chrome://tracing) or at
  [https://ui.perfetto.dev/](https://ui.perfetto.dev/).
- **`${label}.trace.json.gz`** — same file, gzipped. Match the shape of the
  historical `03-C-first-paint.json.json.gz`. Check the gz in; drop the raw
  from git (it's 10× larger).
- **`${label}.metrics.json`** — small human-readable summary:

  ```json
  {
    "captured_at": "2026-07-19T...Z",
    "label": "main-baseline",
    "target_url": "http://127.0.0.1:5600/?path=/.../corpus",
    "viewport": { "width": 1440, "height": 900, "device_scale_factor": 2 },
    "wait_after_nav_ms": 5000,
    "metrics": {
      "lcp_ms": 1561,
      "fcp_ms": 420,
      "ttfb_ms": 40,
      "long_tasks_count": 6,
      "long_tasks_total_ms": 850,
      "memory": { "jsHeapUsedMB": 180, "jsHeapLimitMB": 4096 }
    },
    "trace_events_count": 42315
  }
  ```

Only the `.metrics.json` is checked in for casual comparison. Full traces
land under the same dir but are only needed for deep investigation.

## 5. Fair-comparison invariants

For numbers to be comparable across runs:

- **Same corpus.** Node count dominates LCP. `prod-v2` is the canonical
  reference (833 nodes at the time of Tier-C 1561ms).
- **Same viewport.** Default `1440×900 @ DPR-2` matches the Tier-C recording.
  Change deliberately, not accidentally.
- **Same machine state.** Close heavy Docker / video / model-inference
  workloads before capture. The script does not enforce this — the operator
  must.
- **Multiple runs.** Chromium's LCP fluctuates ±10% run-to-run under
  identical inputs. Capture at least 3 traces per condition, take the
  median, note the range.

## 6. Troubleshooting

**"api port 8600 already in use"** — another instance still running or a
previous run's cleanup didn't complete. `lsof -iTCP:8600 -sTCP:LISTEN` shows
the holder. Kill it, or pass `--api-port <free-port>`.

**"api did not become healthy in 30s"** — check `$API_LOG` printed on
teardown. Common cause: `--corpus` path exists but has no artifacts (empty
dir), api starts but health check fails. Rebuild the corpus.

**"viewer did not become healthy in 30s"** — check `$VIEWER_LOG`. Common
cause: node module resolution failure. Try
`cd web/gi-kg-viewer && env -u NODE_OPTIONS npm ci` in the failing worktree.

**LCP number wildly different from expected** — did the corpus enrichment
artifacts land? A prod-v2 corpus missing `topic_theme_clusters.json` renders
without the theme-region overlay, which changes LCP. Check
`ls $CORPUS/enrichments/`.

**"Cannot find module 'restore-node-options.cjs'"** — cmux/agent-runtime
NODE_OPTIONS poison. Every `node` / `npm` invocation in the script uses
`env -u NODE_OPTIONS`; if you hit this outside the script, the same fix
applies.

## 7. What "counts" for the baseline

For any perf claim on a PR:

- **The metric is LCP.** FCP is too early (structural-shell paint); TTI too
  variable to compare across runs.
- **The corpus is prod-v2** (833 nodes) unless the change specifically
  targets other node-count regimes.
- **Report median-of-3.** Not one-shot; not the fastest of five.
- **Report the delta with sign.** "1561 → 1420 ms (−9%)" or "1561 → 1740 ms
  (+11%, regression)".
- **Include the `.metrics.json` in the commit** so the number can be
  audited without re-running.

## 8. Related

- `docs/guides/GRAPH_VISUALIZATION_GUIDE.md` §Layout — describes what LCP is
  measuring (the graph settling under fcose after artifact fetch + merge).
- `docs/wip/graph-v3/SUMMARY.md` §Perf — the Tier-C baseline number lives
  there. First entry above.
- `web/gi-kg-viewer/src/utils/cyCoseLayoutOptions.ts` — fcose params. The
  most common target for capture-driven tuning.
- `docs/wip/graph-v3/traces/` — canonical dir. Every checked-in trace lives
  here.
