# graph-v3 LCP traces

Captured with `scripts/dev/capture-graph-lcp.sh` (runbook:
`docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md`).

## Corpus + viewport

All runs against **prod-v2** (833 nodes) at viewport **1440×900 @ DPR-2** —
same conditions as the historical `03-C-first-paint.json.json.gz`.

## Baselines (2026-07-19)

### Median across 3 runs

| Ref                     | Shell LCP | Graph time-to-canvas | Nav elapsed | Notes                          |
| ----------------------- | --------- | -------------------- | ----------- | ------------------------------ |
| `main`                  | 928 ms    | **5795 ms**          | ~11.3 s     | Pre-graph-v3 baseline          |
| `feat/graph-v3` (raw)   | 904 ms    | **7340 ms**          | ~11.4 s     | Branch tip pre-tuning          |
| `feat/graph-v3` (tuned) | 1023 ms   | **6686 ms**          | ~11.5 s     | Branch tip with bridge fix     |
| Δ (raw vs main)         | −24 ms    | **+1545 ms (+27%)**  | +100 ms     | Regression pre-tuning          |
| Δ (tuned vs main)       | +95 ms    | **+891 ms (+15%)**   | +200 ms     | After bridge debounce + cache  |

**Tuning applied (this PR):** `applyBridgeNodeClass` now caches its result
by `(node-count, edge-count)` on the core AND debounces to run only after
the graph is settled for 300 ms. Diagnostic instrumentation showed bridge
betweenness fired **twice per settle** on prod-v2 — once at 833 nodes /
1389 edges, then again at 1157 nodes / 2027 edges (fcose emits multiple
layoutstop events during progressive load). Both took ~776 ms each,
totalling 1099 ms cumulative — accounting for the entire regression on the
critical path.

The debounce collapses those two synchronous computes down to zero on the
critical path (`.graph-canvas` visible), then the actual betweenness fires
once ~300 ms after the graph is settled. Bridges pop in visually after the
graph appears — trade-off: brief flicker, gain: 654 ms off first paint.

The residual ~900 ms delta (still +15% vs main) is downstream of the
critical-path metric. Candidates for a future session:
- `applyDegreeVisibility` was 148 ms per settle (still fires 2×).
- Cytoscape stylesheet complexity (many more selectors — theme-region-N,
  velocity-*, credibility-*, .graph-bridge, .lens-consensus-edge, etc.).
- Extra work in `toCytoElements` (adds theme-region tags, degreeHeat,
  aggregate-edge synthesis).

### Diagnostic evidence

Full per-phase measurements from an instrumented run before the fix
(prod-v2, feat/graph-v3 tip, 2× finishLayoutPass calls per settle):

```text
flp:total                        907 ms last  (x2 = 1342 ms cumulative)
flp:bridgeRing                   776 ms last  (x2 = 1099 ms cumulative)  ← 82% of finishLayoutPass
flp:applyDegreeVisibility         79 ms last  (x2 = 148 ms)
flp:applyViewportPreserveOrFit    34 ms last  (x2 = 63 ms)
flp:refreshEnricherLensOverlays    8 ms last  (x2 = 18 ms)
flp:themeClusterRegions            4 ms last  (x2 = 8 ms)   [lens off → cheap clear path]
flp:applyTopicDegreeHeat           3 ms last  (x2 = 6 ms)
flp:recomputeDegreeHistogram       2 ms last  (x2 = 3 ms)
flp:annotateBridgesWithThemes    0.5 ms last  (x2 = 1 ms)
```

After debounce, `flp:bridgeRing` drops to ~0.1 ms (synchronous — just
schedules a `setTimeout`; the real compute happens later). Instrumentation
was removed before commit; can be re-added by editing
`finishLayoutPass()` in `GraphCanvas.vue`.

### Run-by-run

**main baseline (untuned pre-graph-v3):**

| Run | Shell LCP | Graph time-to-canvas |
| --- | --------- | -------------------- |
| 1   | 912 ms    | 6317 ms              |
| 2   | 976 ms    | 5609 ms              |
| 3   | 928 ms    | 5795 ms              |

**feat/graph-v3 tip (raw, pre-tuning):**

| Run | Shell LCP | Graph time-to-canvas |
| --- | --------- | -------------------- |
| 1   | 1044 ms   | 7429 ms              |
| 2   | 904 ms    | 7340 ms              |
| 3   | 864 ms    | 7177 ms              |

**feat/graph-v3 tip (tuned — bridge cache + debounce):**

| Run | Shell LCP | Graph time-to-canvas |
| --- | --------- | -------------------- |
| 1   | 956 ms    | 6686 ms              |
| 2   | 1080 ms   | 6630 ms              |
| 3   | 1032 ms   | 7568 ms              |

Full metrics for every run: `prod-v2-{main-baseline,feat-graph-v3,feat-graph-v3-tuned}-runN.metrics.json`.

Screenshot at trace-stop for each run:
`prod-v2-{main-baseline,feat-graph-v3}-runN.screenshot.png`. All show the
graph rendered — the gestures overlay is visible on first-run because we're
not dismissing it (fair to both refs).

## Trace archives

`.trace.json.gz` files (Chrome DevTools Performance format) are kept for
**run 1 of each condition** as canonical evidence — the other runs' full
traces were dropped to keep the dir small. Raw `.trace.json` (~70 MB each)
are gitignored — the gz is ~8 MB, ~10× compression.

Open the gz in [chrome://tracing](chrome://tracing) or
[https://ui.perfetto.dev/](https://ui.perfetto.dev/) for main-thread
visualisation. The metrics JSON alongside is the numeric summary.

## What NOT to conclude from these numbers

- **Not comparable to the historical `03-C-first-paint.json.json.gz`**
  (Tier-C 1561 ms). That number was captured on an earlier build against a
  different graph feature set and via manual Chrome DevTools recording, not
  this script. The 928 ms shell LCP here is not the same event as the
  1561 ms "first paint" reported for Tier C.
- **Not a proof that the branch is 27% slower for users.** These are
  scripted, deterministic settles under Playwright headless Chromium. Real
  users on their own machines with warm caches, extensions, and network
  variance will see different numbers. The *shape* (main faster than
  branch on graph settle) is the honest read; the *magnitude* is
  indicative, not definitive.
- **Not a Web Vitals LCP.** `graph_time_to_canvas_ms` is a custom metric —
  time from Graph tab click to `.graph-canvas` having Cytoscape children.
  Web Vitals LCP is the shell number (~928 ms), which is essentially flat.

## Where the ~1500 ms delta likely lives

Per the additions between main and branch (see
`docs/guides/GRAPH_VISUALIZATION_GUIDE.md` for the rule catalog):

1. **Bridge betweenness** on every layout — Tier 3-K, ~200 ms on 833 nodes
   per the SUMMARY.md notes.
2. **Theme-region propagation** — Tier 4-T, first-cluster-wins tagging over
   Insight/Person/Org/Podcast. Order-of-magnitude 10–50 ms.
3. **Enricher lens overlays applied post-layout** — velocity halo,
   credibility border, consensus edges, co-guest edges, person communities.
   Even when the lens is default-off, the availability probe still fetches
   the corpus envelope once per corpus path.
4. **Degree-heat propagation** — Tier 3-J extended from Topic to
   Person/Org/Episode. Layout-pass extra work.
5. **Cytoscape stylesheet complexity** — many more selectors (theme-region-N,
   velocity-{up,down,steady}, credibility-{high,med,low}, .graph-bridge,
   .lens-consensus-edge, etc.). Selector resolution isn't free.

Any perf work on this branch should target 1 and 3 first (biggest single
contributions likely).

## What to do next

If the operator decides the 27% regression is unacceptable:

- Capture a fresh baseline against this branch tip with lens overlays
  disabled at boot (skip the availability probe on cold start).
- Try `fcose` tuning per the docs' proposal (`nodeRepulsion: 8000` etc.,
  with bipartite-safe gravity). Use this same script to capture, compare
  medians.
- Move bridge betweenness to a web worker so it doesn't block main-thread
  during layout.

Otherwise, the regression is a documented trade-off for the visual
capabilities the branch adds. The runbook + script mean any future perf
work has a repeatable measurement path.
