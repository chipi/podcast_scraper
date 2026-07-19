# Graph tech debt

Running log of graph-viewer improvements that were surfaced during work
but were **out of scope** for the PR that surfaced them. When we tackle
these, this file is the launching pad — each item has enough context
that a future session (or a new agent) can pick it up cold.

Add items with:

```markdown
## <short title>

**Surfaced by:** PR # / commit / session date
**Category:** perf | correctness | UX | refactor | test-harness
**Estimated cost:** S (< 4h) / M (4h–2d) / L (> 2d)
**Blocking:** what's blocked on this fix, if anything
**Context:** what's the concrete finding + reproduction path
**Why deferred:** why not in the surfacing PR
**Handle:** the specific file / function / config to change

Related: cross-refs (issues, RFCs, other WIP notes).
```

Order by category then by cost. Keep the list scanned. When an item
lands, move its subsection here to a "Resolved" section at the bottom
with a link to the PR that resolved it, so the file stays a diary
of what was addressed and why.

---

## Perf — wave 1 fcose is the residual cost centre on graph settle

**Surfaced by:** PR #1207 / #1211 / 2026-07-19
**Category:** perf
**Estimated cost:** M — L (depends on approach)
**Blocking:** closing the last +408 ms (+7%) regression vs main on prod-v2 graph settle
**Context:** PR #1207's perf work landed 1137 ms of savings (bridge cache/debounce + #1211 fast path + fcose label opt-out), leaving a residual +408 ms (+7%) on graph time-to-canvas vs main. Diagnostic in
[`docs/wip/graph-v3/traces/README.md`](graph-v3/traces/README.md) attributed the residual to wave 1's fcose layout on 833 nodes on prod-v2 — now the single largest cost on the critical path.

Empirical tuning (documented in the traces README + #1211 comment) tried
every obvious fcose lever without breaking layout: numIter, nodeRepulsion,
nodeDimensionsIncludeLabels. `quality: 'draft'` saved 1465 ms but broke
layout (one dense clump, empty half of canvas — screenshot at
`/tmp/lcp-diag/prod-v2-draft-run3.screenshot.png` at the session).

**Why deferred:** further gains require a bigger refactor than a single PR should absorb.
**Handle — three candidate approaches:**

- **Move fcose to a web worker.** Breaks the current main-thread contract (`layoutstart` / `layoutstop` events synchronous from the operator's perspective). Would need a shim that proxies fcose progress. Estimated L.
- **Progressive rendering.** Show the graph at an initial "just-positioned" state (no fcose relaxation, nodes near neighbours) with an animated transition to the settled layout as fcose completes. Real UX change; would need design review. Estimated M.
- **Alternative layout algorithm.** Explore cola / dagre / other — unknown cost/quality trade-off; would need a full head-to-head evaluation.
  Estimated M.

**Measurement path:** `scripts/dev/capture-graph-lcp.sh` +
`docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md` are the reusable measurement
contract landed in PR #1207. Median-of-3 on prod-v2 is the metric.

**Related:** #1211 (delivered narrow-scope fast path), #1207 (PR context), `docs/wip/graph-v3/traces/README.md` (diagnostic evidence + tuning session log).

---

## Perf — topDown load mode still parses full artifact on initial mount

**Surfaced by:** HD22 topDown baseline capture / 2026-07-19
**Category:** perf
**Estimated cost:** M — L
**Blocking:** flipping the default to `'topDown'` giving the multi-second win the design intent implied
**Context:** HD22 median-of-3 shows `topDown` mode only saves 900 ms
(14%) vs `everything` on prod-v2 initial mount — 5328 ms vs 6228 ms.
Small win for a 6-node vs 833-node canvas.

Root cause: `web/gi-kg-viewer/src/stores/artifacts.ts:126` — the
`topDownDisplayArtifact` computed derives its slice from
`displayArtifact.value?.data`, so the full corpus envelope must fetch
and the full merged artifact must parse before the 6-SuperTheme slice
can be projected. `topDown` is a **display-side filter**, not a
data-fetch short-circuit.

Evidence + full write-up:
[`traces/README.md § TopDown load-mode audit`](graph-v3/traces/README.md#topdown-load-mode-audit-2026-07-19).

**Why deferred:** the fix is an `artifacts.ts` refactor that would let
topDown skip the full-artifact parse (fetch theme_clusters only, lazy
resolve child artifacts on expand). Real architectural work + test
coverage — not something to smuggle into the current PR.

**Handle — candidate designs:**

- **Lazy-load per-super-theme child artifacts on expand-on-tap.** On
  initial mount, `topDownDisplayArtifact` builds from
  `themeClustersDoc` alone (already exists). On tap, fetch only that
  super-theme's contained artifacts. Cheapest cold load. Cost: rework
  the `displayArtifact` computed to be optional in topDown mode + the
  expand handler to trigger targeted fetches.
- **Idle-preload of the full artifact.** Show topDown immediately;
  fetch/parse the full artifact in background so expand-on-tap has
  the data ready without a network roundtrip. Middle ground —
  keeps the current `displayArtifact` shape, just moves the fetch
  off the critical path.

**Measurement path:** `scripts/dev/capture-graph-lcp.sh --load-mode
topDown`. See `docs/guides/GRAPH_PERF_TRACE_RUNBOOK.md`.

**Related:** #1207, `graph-tech-debt.md` wave-1 fcose item above,
`docs/wip/graph-v3/SUMMARY.md § Tier 8`.

---

## Perf — expand-on-tap rebuilds cy from scratch (100+ nodes per tap)

**Surfaced by:** HD22 expand-on-tap probe / 2026-07-19
**Category:** perf
**Estimated cost:** M
**Blocking:** topDown mode feeling snappy on interaction
**Context:** HD22 expand-on-tap probe shows every SuperTheme tap
grows the slice by ~100 nodes (6 → 106 on prod-v2 for `sth:interest-rates`),
and the current handler destroys + recreates the cytoscape core on
each tap — same pattern as the KG-second-wave rebuild called out in
the wave-1 fcose item. Wall time per expand hit our 4000 ms
instrumentation ceiling every run.

**Why deferred:** same root cause as the wave-1 fcose residual — the
proper fix is `cy.add(delta)` instead of full rebuild, which is a
real refactor. HD22 confirms the fix applies to both critical paths
(cold load AND expand-on-tap), so it's a single unified refactor,
not two.

**Handle:** the fast-path from #1211
(`GraphCanvas.vue::tryIncrementalAppendFastPath`) is the seed
implementation for the `everything` mode's incremental append. The
expand-on-tap handler needs the same shape: detect
"deltaNodes ⊂ cy AND deltaNodes ⊃ cy" and take the fast path when
true.

**Measurement path:** `scripts/dev/capture-graph-lcp.sh
--load-mode topDown --expand-first-super-theme`. Re-instrument
`finishLayoutPass` with `performance.measure('flp:total', ...)` to
break the 4000 ms ceiling and see the real expand cost.

**Related:** wave-1 fcose item above, HD22 audit.

---

## Perf — `quality: 'draft'` fcose is a documented off-lever

**Surfaced by:** #1211 tuning session / 2026-07-19
**Category:** perf
**Estimated cost:** S (revert + measure)
**Context:** Setting `quality: 'draft'` in `web/gi-kg-viewer/src/utils/cyCoseLayoutOptions.ts` cut graph settle time from 6352 ms to 4887 ms on prod-v2 — a **1465 ms (~30%) saving**. But layout quality degraded: one dense clump on the right + empty left half. Unusable on prod-v2.

**Why deferred:** the visual regression outweighs the perf win on the corpora we have today. A different corpus (denser, more evenly connected) might tolerate it.

**Handle:** flag location is documented at the assignment site in
`web/gi-kg-viewer/src/utils/cyCoseLayoutOptions.ts`. To try again on a
new corpus: flip to `'draft'`, capture with the LCP tool, compare
median + visually inspect the settle screenshot.

---

*More items land here as they're surfaced. See git history of this file for the diary of what was addressed.*
