# graph-v3 LCP traces

Captured with `scripts/dev/capture-graph-lcp.sh` (runbook:
`docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md`).

## Corpus + viewport

All runs against **prod-v2** (833 nodes) at viewport **1440×900 @ DPR-2** —
same conditions as the historical `03-C-first-paint.json.json.gz`.

## Baselines (2026-07-19)

### Median across 3 runs

| Ref                       | Shell LCP | Graph time-to-canvas | Nav elapsed | Notes                              |
| ------------------------- | --------- | -------------------- | ----------- | ---------------------------------- |
| `main`                    | 928 ms    | **5795 ms**          | ~11.3 s     | Pre-graph-v3 baseline              |
| `feat/graph-v3` (raw)     | 904 ms    | **7340 ms**          | ~11.4 s     | Branch tip pre-tuning              |
| `feat/graph-v3` (bridge)  | 1023 ms   | **6686 ms**          | ~11.5 s     | Bridge fix only                    |
| `feat/graph-v3` (fastpath)| 1016 ms   | **6305 ms**          | ~11.5 s     | Bridge fix + #1211 fast path       |
| `feat/graph-v3` (labels)  | 1032 ms   | **6203 ms**          | ~11.5 s     | + nodeDimensionsIncludeLabels:false|
| Δ raw vs main             | −24 ms    | **+1545 ms (+27%)**  | +100 ms     | Regression pre-tuning              |
| Δ bridge vs main          | +95 ms    | **+891 ms (+15%)**   | +200 ms     | After bridge debounce + cache      |
| Δ fastpath vs main        | +88 ms    | **+510 ms (+8.8%)**  | +200 ms     | After #1211 cy-anchor fast path    |
| **Δ labels vs main**      | +104 ms   | **+408 ms (+7%)**    | +200 ms     | **After fcose label opt-out**      |

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

The residual ~900 ms delta (still +15% vs main) was traced to
**the graph being fully rebuilt twice** during initial load — the
"KG-second-wave" pattern from GH #771. A separate diagnostic run
(marks `rdw:*` in `redraw()`, since removed) attributed the pipeline
to:

| Phase                       | Time        | Contributing to residual?                              |
| --------------------------- | ----------- | ------------------------------------------------------ |
| `cy` init (redraw 1)        | 233 ms      | Yes — cy created twice                                 |
| **fcose layout (redraw 1)** | **1461 ms** | **Yes — 100% thrown away when redraw 2 replaces `cy`** |
| `finishLayoutPass` 1        | 128 ms      | Fixed by debounce                                      |
| Inter-redraw gap            | 989 ms      | Second-wave artifact fetch                             |
| `cy` init (redraw 2)        | 273 ms      | Yes — cy recreated                                     |
| **fcose layout (redraw 2)** | **3109 ms** | Unavoidable on this size                               |
| `finishLayoutPass` 2        | 156 ms      | Fixed by debounce                                      |

**Redraw 1's 1461 ms of fcose is discarded** when the KG-second-wave
arrives ~1 second later and `cytoscape({elements})` is called again with
the superset. That's the residual.

The proper fix is in the redraw handler: when
`isIncrementalAppend` is true (already detected — see
`GraphCanvas.vue::filteredArtifact` watcher L4054), instead of destroying
and recreating cy, use `cy.add(deltaElements)` and preserve the existing
fcose layout for old nodes. The current code paths (selection preservation,
FSM handoff, camera preservation) all assume full-rebuild semantics, so
this is a real refactor that deserves its own PR + test coverage — not
rushed into this one. Candidate savings: ~1200–1400 ms off first paint.

Debounce tuning alone cannot fix this — the second wave arrives *after*
redraw 1 completes, not during the debounce window. Delaying redraw 1
enough to catch redraw 2 would mean ~1500 ms of blank tab, worse UX than
the current two-pass render.

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

+ **Not comparable to the historical `03-C-first-paint.json.json.gz`**
  (Tier-C 1561 ms). That number was captured on an earlier build against a
  different graph feature set and via manual Chrome DevTools recording, not
  this script. The 928 ms shell LCP here is not the same event as the
  1561 ms "first paint" reported for Tier C.
+ **Not a proof that the branch is 27% slower for users.** These are
  scripted, deterministic settles under Playwright headless Chromium. Real
  users on their own machines with warm caches, extensions, and network
  variance will see different numbers. The *shape* (main faster than
  branch on graph settle) is the honest read; the *magnitude* is
  indicative, not definitive.
+ **Not a Web Vitals LCP.** `graph_time_to_canvas_ms` is a custom metric —
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

If the operator decides the 15% residual is unacceptable:

+ Land the "no rebuild on incremental append" refactor described above —
  the biggest handle and the actual root cause. ~1200–1400 ms saving.
+ Move bridge betweenness to a web worker so it doesn't block main-thread
  during layout. Lower impact now that we debounce it, but still non-zero.
+ Try `fcose` tuning per the docs' proposal (`nodeRepulsion: 8000` etc.,
  with bipartite-safe gravity). Use this same script to capture, compare
  medians.

Otherwise, the regression is a documented trade-off for the visual
capabilities the branch adds. The runbook + script mean any future perf
work has a repeatable measurement path.

## TopDown load-mode audit (2026-07-19)

The `graphLoadMode: 'topDown'` opt-in (tier 8, opt-in via the
Load-mode chip, default is still `'everything'`) mounts a synthetic
6-SuperTheme rollup instead of the 833-node full graph. The perf
question: how much does that save, and where's the residual cost?

**Setup:** identical to sections above — prod-v2, 1440×900 @ DPR-2,
median-of-3, feat/graph-v3 tip. LocalStorage seeded to
`ps_graph_load_mode=topDown` before navigation via
`context.addInitScript` (see `scripts/dev/capture-graph-lcp.mjs`
`LCP_LOAD_MODE`).

### Median-of-3 (feat/graph-v3 tip, 2026-07-19)

| Load mode                | ttcanvas  | Δ vs `main` | Notes                              |
| ------------------------ | --------- | ----------- | ---------------------------------- |
| `main` (historical)      | 5795 ms   | —           | Pre-graph-v3 reference             |
| `everything` (re-baseline)| **6228 ms** | +433 ms (+7.5%) | This-branch tuned everything mode  |
| `topDown` (rollup)       | **5328 ms** | −467 ms (−8%)   | 6-SuperTheme mount, no expand |
| **Δ topDown vs everything** | **−900 ms (−14%)** | — | The load-mode savings on this branch |

Per-run detail:

| Load mode    | run1    | run2    | run3    | median  |
| ------------ | ------- | ------- | ------- | ------- |
| `topDown`    | 5582 ms | 5328 ms | 4862 ms | 5328 ms |
| `everything` | 6218 ms | 6312 ms | 6228 ms | 6228 ms |

### Finding: topDown does NOT skip the full artifact parse

Only 900 ms savings for a 6-node vs 833-node canvas is a strong signal
that fcose is **not** the dominant cost on the initial mount. Root cause:
`stores/artifacts.ts:126` — `topDownDisplayArtifact` derives from
`displayArtifact.value?.data`, so the full corpus envelope fetch + full
merged-artifact parse must complete before the 6-SuperTheme slice can
be projected. TopDown is a **display-side filter**, not a data-fetch
short-circuit.

Where the ~5.3 s topDown ttcanvas is spent (order of magnitude, no
per-phase instrumentation currently active — inferred from the
`everything`-mode diagnostic table earlier in this doc + the fact that
fcose on 6 nodes is essentially free):

- **~800–1000 ms** shell + Graph tab click + FSM warmup.
- **~2000–3000 ms** corpus envelope fetch + full artifact parse
  (same in both modes).
- **~500 ms** cy init + tiny fcose settle on 6 nodes.
- **~500–1000 ms** finishLayoutPass housekeeping (theme-region
  regions, degree histogram, viewport-preserve).

The 900 ms delta vs `everything` is roughly the fcose-on-833-nodes
plus bridge betweenness savings — real, but not the multi-second win
you'd expect just from node count. **Flipping the default to
`'topDown'` therefore does not solve the perf problem** — the
critical-path cost is upstream of fcose.

### Expand-on-tap probe (feat/graph-v3 tip, 2026-07-19)

The mjs script grew an `LCP_EXPAND_FIRST_SUPERTHEME=1` mode that taps
the first SuperTheme node after canvas mount and measures resettle. The
target SuperTheme (`sth:interest-rates` on prod-v2, deterministic
"first" by cy iteration order) expands the slice from 6 → 106 nodes in
one tap.

**First pass — instrumentation OFF (fallback ceiling):**

| Run   | Initial ttcanvas | Expand click-to-settle |
| ----- | ---------------- | ---------------------- |
| run 1 | 4731 ms          | ≥ 4000 ms (timeout)    |
| run 2 | 6950 ms          | ≥ 4000 ms (timeout)    |
| run 3 | 4590 ms          | ≥ 4000 ms (timeout)    |

The 4000 ms figure was our fallback ceiling — the mjs waits for a
`performance.measure('flp:total', ...)` from `finishLayoutPass()`,
and if it never arrives the promise resolves at the ceiling. On the
shipped build with `flp:total` instrumentation removed, every expand
run hit the ceiling — but that tells us nothing about the real
settle time.

**Correction (HD23 instrumentation-on rerun, prod-v2, 2026-07-19):**

`finishLayoutPass` now always emits `performance.mark('flp:start:<n>')`
+ `flp:end:<n>` + `flp:total` measures (see `GraphCanvas.vue` around
line 1834 and its tail). Re-running the same probe:

| Run   | Initial ttcanvas | Expand click-to-settle | `flp:total` last | `flp:total` calls |
| ----- | ---------------- | ---------------------- | ---------------- | ----------------- |
| run 1 | 5497 ms          | **109 ms**             | 61 ms            | 2                 |

**The expand-on-tap fast path was already firing.** The 4000 ms
"ceiling" in the earlier table was pure measurement noise. #1211's
`tryIncrementalAppendFastPath` catches the SuperTheme-tap →
`topDownDisplayArtifact` recompute → cy-anchored superset check
correctly, appends the ~100 delta nodes near their existing
neighbours, and skips fcose entirely. Total settle: ~109 ms
(finishLayoutPass 61 ms + housekeeping).

**No refactor needed for expand-on-tap.** The `graph-tech-debt.md`
entry that flagged it is closed out.

### KG-second-wave probe (feat/graph-v3 tip, `everything` mode, 2026-07-19)

With `flp:total` instrumentation on:

| Run   | ttcanvas | `flp:total` last | `flp:total` calls |
| ----- | -------- | ---------------- | ----------------- |
| run 1 | 6697 ms  | 212 ms           | **2**             |

Two `flp:total` calls confirms the KG-second-wave still does a full
rebuild — the fast path bails. Root cause is upstream of the render
pipeline: the GI→KG merger in
`web/gi-kg-viewer/src/utils/mergeGiKg.ts:219-243` (`remapData`)
renames **every** GI node on wave 2 — id `X` becomes `g:X`, KG nodes
become `k:X`. Wave-1 cy has ids like `ep:X`, `topic:Y`, `person:Z`
raw; wave-2's artifact has the same nodes as `g:ep:X`, `g:topic:Y`,
`g:person:Z`. The strict-superset check (`filteredArtifact` watcher
in `GraphCanvas.vue::4176-4204`) correctly returns false, and cy
gets destroyed + rebuilt.

Potential savings if the fast path fires on wave 2: the ~3109 ms
wave-2 fcose settle (from the earlier diagnostic table) becomes a
~200-400 ms delta-add. Roughly **2500–3000 ms off `everything` cold
load**.

Fixing this requires either:

1. **Merger canonicalisation on wave 1.** Apply `g:` prefix to every
   GI node id even when KG hasn't loaded, so wave-2's remap becomes
   a no-op for those nodes. Cross-cutting consumer audit needed —
   every code path that reads raw ids from a wave-1 artifact (bridge
   doc, enricher artifacts, focus resolution, search, subject-rail,
   URL routing, topic-cluster mapping) has to accept the prefixed
   form.
2. **Fast path with id-transform awareness.** Teach
   `tryIncrementalAppendFastPath` the wave-1 → wave-2 transform (`X
   → g:X` etc.) so the superset check succeeds despite the rename,
   and the delta-add applies remove+add for renamed nodes with
   preserved positions. Surgical, but the render pipeline has to
   know about the merger.

Both are larger than a scoped PR on top of `feat/graph-v3`.
Deferred to a follow-up issue — this branch ships instrumentation
+ audit only.

**What we DO know from the probe:**

- Every expand adds **100 nodes** to cy (6 → 106). The current code
  destroys + recreates cy for each expand (same code path as the
  KG-second-wave rebuild), so we pay the full fcose settle on 106
  nodes every tap. That is directly measurable: fcose on 100 nodes
  is not free — the trace README shows fcose on 833 nodes as
  ~1461 ms + ~3109 ms in the two-wave case; a proportional
  estimate for 106 nodes is ~200–400 ms of fcose alone.
- The 4000 ms wall we hit suggests the rest is Vue reactivity
  cascades from filter re-scope + FSM handoff + theme-region
  repaint on the enlarged slice — the same finishLayoutPass work
  the `everything` mode pays on every layoutstop, now paid per
  expand.

To break this ceiling we need one of:
1. Re-instrument `finishLayoutPass` (small revert) to see the
   true expand cost.
2. `cy.add(delta)` on expand instead of full rebuild — same
   refactor as the KG-second-wave fix.
3. Both.

### Where the fcose tuning work stands after HD22

HD22 delivered the measurement contract for the topDown critical
path. What it made visible:

- **Initial mount:** fcose is NOT the bottleneck for topDown.
  Corpus-fetch + full-artifact-parse dominate. A real fcose wave-2
  focused on the initial mount would need to *also* shortcut the
  full-artifact parse when in topDown mode. That's an
  `artifacts.ts` refactor, not an fcose-options tune.
- **Expand-on-tap:** fcose IS a real cost (100-node settle per
  tap). The `cy.add(delta)` refactor called out in
  `docs/wip/graph-tech-debt.md:31` applies here just as much as
  to the KG-second-wave `everything` path.

**Net:** fcose tuning that's just about options in
`cyCoseLayoutOptions.ts` is essentially done for the `everything`
critical path (labels off, gravity 0.12, numIter cap, `quality:
'draft'` documented as off-lever). The remaining wins require
architectural work — either shortcut the parse for topDown, or
land `cy.add(delta)` incremental append. Both belong in a scoped
PR of their own; see the follow-up items in `graph-tech-debt.md`.

### Reproducing

```
scripts/dev/capture-graph-lcp.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label prod-v2-topdown-runN \
  --load-mode topDown \
  --wait-ms 6000

# expand probe (requires dev build so window.__GIKG_CY_DEV__ is exposed):
scripts/dev/capture-graph-lcp.sh \
  --corpus .test_outputs/manual/prod-v2/corpus \
  --label prod-v2-topdown-expand-runN \
  --load-mode topDown \
  --expand-first-super-theme \
  --wait-ms 8000
```

`--load-mode everything` seeds the localStorage key too, so the
`everything` re-baseline is directly comparable.
