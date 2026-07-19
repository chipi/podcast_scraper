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
