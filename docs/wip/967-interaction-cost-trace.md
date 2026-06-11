# #967 — Large-graph interaction-cost trace (fcose era)

**Date:** 2026-06-11
**Context:** After the cose→fcose swap (#967 issue 1) the O(n²) layout freeze is gone, so
the `GRAPH_DEFAULT_EPISODE_CAP` no longer needs to bound *layout* time. This trace answers
the follow-up: **what is the real ceiling now, and how high can the cap go?**

## Method

Live viewer (`make serve-for-validation`, vite dev) driving the real `cytoscape` instance via
the dev hook `window.__GIKG_CY_DEV__`. Loaded the synthetic validation corpus graph (117 nodes,
real node classes/styles), then synthetically grew the live graph (cloning real node classes so
styling + labels render representatively) and at each size measured:

- **fcose layout** — `cy.elements().layout({name:'fcose', animate:false}).run()`, time to `layoutstop`.
- **Interaction** — a 140-frame `requestAnimationFrame` loop forcing continuous `panBy` + periodic
  `zoom` (worst-case: every frame repaints + re-hit-tests), recording per-frame deltas → p50/p95/max.

**Hardware caveat:** Apple Silicon-class laptop, **14 cores, `devicePixelRatio = 1`**. FPS is
device- and DPR-dependent: a retina display (DPR 2) quadruples pixel-fill, and weaker devices
degrade sooner. Treat these as an *optimistic* ceiling for typical hardware.

## Results

| Nodes | ≈Episodes (prod density ~28 nodes/ep) | fcose layout | pan/zoom p50 | p95 | max | ≈FPS | Verdict |
|------:|--------------------------------------:|-------------:|-------------:|----:|----:|-----:|---------|
| 117   | ~4   | 86 ms   | 17 ms | 18 ms  | 18 ms  | **60** | smooth (vsync-capped) |
| 1500  | ~53  | ~0.9 s  | 33 ms | ~50 ms | 67 ms  | **~37** | good |
| 2117  | ~75  | ~1.5 s  | 33 ms | ~70 ms | 100 ms | **~25** | sluggish but usable |
| 2883  | ~100 | 2.4 s   | 50 ms | 67 ms  | 117 ms | **~20** | laggy |

(The synthetic 1500-node real-browser layout ~0.9 s matches the headless perf test in
`cyFcoseLayout.test.ts`, confirming the measurement.)

## Findings

1. **Layout is no longer the limiter.** fcose settles even ~2.9k nodes in ~2.4 s (cose was ~134 s
   at the same size — a >50× win). It scales sub-quadratically and never freezes the tab.
2. **Interaction (pan/zoom repaint) is the new limiter**, degrading ~linearly with node count:
   60 fps → ~37 fps (1.5k) → ~25 fps (2.1k) → ~20 fps (2.9k). p95 frame time crosses the
   ~50 ms (20 fps) "feels janky" line around ~2k nodes.
3. **`GRAPH_DEFAULT_EPISODE_CAP = 50` (≈1.4k nodes → ~37 fps) is the right *smooth* default.**
   ~75 ep (~2.1k) is the tolerable edge; 100 ep (~2.9k) is laggy *on capable hardware* and would
   be worse on retina / low-end devices.

## Recommendation

- **Keep the default cap at 50.** It is the empirically-validated smooth ceiling, not a guess.
- **Do NOT raise the default toward the full corpus** without first adding interaction-cost
  mitigations, because retina/low-end devices hit the wall well before 2.9k nodes. Candidate
  mitigations before any future raise:
  - Level-of-detail during pan/zoom (hide labels + edges mid-gesture, restore on settle).
  - Edge culling / bundling at low zoom (edges dominate repaint cost).
  - An explicit opt-in "load all episodes (may be slow)" affordance instead of a higher default.

## Side finding (separate from #967) — API segfault on a big legacy-schema corpus

While loading `.test_outputs/manual/my-manual-run-10` (12,550 vectors, lance index built before
the schema_version=2 change), the serve-api process **segfaulted (`make: *** [serve-api] Error 139`
= SIGSEGV)**. Sequence in the log:

```
GET /api/corpus/feeds … 200 OK
hybrid_search: lance index … has a stale schema (v1 < v2); falling back to FAISS — rebuild via `cli index-two-tier`
<segfault>
```

- The **lance schema-staleness guard worked as designed** (correctly flagged v1 < v2 and fell back).
- The crash is a **native SIGSEGV in the FAISS fallback path on a large legacy index** — not a
  viewer/#967 bug, and not caused by the staleness change (it only routed to the pre-existing
  FAISS path). Worth filing a separate robustness issue + confirming reproducibility; a stale-index
  corpus should be rebuilt (`cli index-two-tier` / `cli upgrade`) rather than crash the API.
