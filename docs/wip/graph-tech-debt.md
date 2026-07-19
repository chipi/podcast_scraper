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

## Perf — KG-second-wave forces full cy rebuild (merger id rename)

**Surfaced by:** HD22 audit + HD23 instrumentation / 2026-07-19
**Category:** perf
**Estimated cost:** M — L (bigger of the two variants below)
**Blocking:** ~2500–3000 ms of `everything`-mode cold-load fcose that gets thrown away
**Context:** HD23 landed `flp:total` instrumentation in
`GraphCanvas.vue::finishLayoutPass`. The `everything`-mode cold load
now shows **2 × flp:total** calls — proof the fast path
(`tryIncrementalAppendFastPath`, #1211) does NOT catch the
KG-second-wave. Cy is destroyed + rebuilt with the merged
wave-2 artifact.

Root cause upstream of the render pipeline:
`web/gi-kg-viewer/src/utils/mergeGiKg.ts:219-243` (`remapData`)
prefixes **every** GI node with `g:` and every KG node with `k:` on
wave 2. Wave-1 cy has raw ids (`ep:X`, `topic:Y`, `person:Z`);
wave-2's artifact has the same nodes as `g:ep:X`, `g:topic:Y`,
`g:person:Z`. Strict-superset check
(`GraphCanvas.vue::filteredArtifact` watcher, lines 4176–4204)
correctly returns false, and cy rebuilds.

**HD22's original write-up UNDERESTIMATED the scope.** It framed
this as a "cy.add(delta) refactor" — but calling `cy.add(delta)`
requires the fast path to recognise wave-1 ids in wave-2's artifact.
The rename hits ~all GI nodes, not just episodes.

**Estimated saving if fixed:** ~2500–3000 ms off `everything`
cold-load (the wave-2 fcose on 1157 nodes gets replaced by a
delta-add + preserved positions).

**Handle — two candidate designs:**

- **Design 1 — Merger canonicalisation on wave 1.** Apply the `g:`
  prefix to every GI node id even when only GI has loaded (wave 1).
  Then wave-2's remap becomes a no-op. Requires a consumer audit —
  every code path that reads raw ids from a wave-1 artifact needs
  to accept the prefixed form. Concrete audit targets: bridge doc,
  enricher artifacts, focus resolution, search, subject-rail node
  lookup, URL routing, `topicClustersOverlay`, `themeClustersOverlay`.
  Estimated M-L.
- **Design 2 — Fast path with id-transform awareness.** Teach
  `tryIncrementalAppendFastPath` (and the superset check) the
  wave-1 → wave-2 transform: `bareId → g:bareId` for the GI half
  plus `__unified_ep__:X` canonicalisation for episodes. On match,
  do targeted remove+add per renamed node with old position
  preserved, and cy.add() the new-only delta. Surgical, but crosses
  merger + render-pipeline concerns. Estimated M.

Either variant needs playwright coverage of the wave-1 → wave-2 → fast-path handoff (production-shaped fixture).

**Measurement path:** `scripts/dev/capture-graph-lcp.sh
--load-mode everything --wait-ms 10000`. Look for `flp:total` calls
in the metrics — should drop from 2 → 1 when the fast path fires.

**Related:** #1211 fast path (seed), HD22 (topDown audit), HD23
(instrumentation + audit correction), GH issue TBD after this PR
lands.

---

## Perf — expand-on-tap fast path already fires (closed by HD23)

**Surfaced + closed by:** HD22 → HD23 / 2026-07-19
**Category:** perf
**Estimated cost:** N/A — no fix needed
**Context:** HD22 flagged expand-on-tap as "≥ 4000 ms per tap
(6 → 106 nodes) rebuilding cy from scratch". That number was pure
measurement noise: the mjs waited for a
`performance.measure('flp:total', ...)` that never fired because
the shipped build had no `flp:total` instrumentation, so the
fallback ceiling of 4000 ms triggered every run.

HD23 added `flp:total` to `finishLayoutPass` and re-ran: real
expand-on-tap is **109 ms** end-to-end, with `flp:total` last =
61 ms. `tryIncrementalAppendFastPath` correctly catches the
SuperTheme-tap → `topDownDisplayArtifact` recompute → cy-anchored
superset check, appends the ~100 delta nodes with pre-positioning,
skips fcose. UX is already good.

Nothing to do. This entry stays as a diary note so a future
reader doesn't chase the same ghost.

**Related:** HD22, HD23, `docs/wip/graph-v3/traces/README.md`
§ Expand-on-tap probe.

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
