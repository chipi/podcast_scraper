# graph-v3 LCP traces

Captured with `scripts/dev/capture-graph-lcp.sh` (runbook:
`docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md`).

## Corpus + viewport

All runs against **prod-v2** (833 nodes) at viewport **1440×900 @ DPR-2** —
same conditions as the historical `03-C-first-paint.json.json.gz`.

## Baselines (2026-07-19)

### Median across 3 runs

| Ref             | Shell LCP | Graph time-to-canvas | Nav elapsed | Notes                          |
| --------------- | --------- | -------------------- | ----------- | ------------------------------ |
| `main`          | 928 ms    | **5795 ms**          | ~11.3 s     | Pre-graph-v3 baseline          |
| `feat/graph-v3` | 904 ms    | **7340 ms**          | ~11.4 s     | Branch tip (this PR)           |
| Δ               | −24 ms    | **+1545 ms (+27%)**  | +100 ms     | Regression on graph settle     |

Interpretation: shell-paint LCP is essentially unchanged (Δ within run-to-run
noise). **Graph time-to-canvas** — click Graph tab → `.graph-canvas` mounted
with Cytoscape children — is **~27% slower on the branch**. The added work
maps directly onto Tier 3 (bridge betweenness on every layout, degree heat
extended to Person/Org/Episode) + Tier 4 (theme-region propagation over
Insight/Person/Org/Podcast) + Tier 5C/5D (enricher lens overlays fetched +
applied post-layout). Each is a documented pass over the graph; the totals
add up.

### Run-by-run

**main baseline (untuned pre-graph-v3):**

| Run | Shell LCP | Graph time-to-canvas |
| --- | --------- | -------------------- |
| 1   | 912 ms    | 6317 ms              |
| 2   | 976 ms    | 5609 ms              |
| 3   | 928 ms    | 5795 ms              |

**feat/graph-v3 tip:**

| Run | Shell LCP | Graph time-to-canvas |
| --- | --------- | -------------------- |
| 1   | 1044 ms   | 7429 ms              |
| 2   | 904 ms    | 7340 ms              |
| 3   | 864 ms    | 7177 ms              |

Full metrics for every run: `prod-v2-{main-baseline,feat-graph-v3}-runN.metrics.json`.

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
