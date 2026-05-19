# Tier-3 real-backend validation walk

**RFC-086 / ADR-095 Tier 3.** Drives a real browser against the running
`make serve` stack with an on-disk corpus the operator picks. Catches
drift between the checked-in `production-shaped` fixture (Tier 2) and the
actual backend / pipeline behaviour.

## How to run

```bash
# 1. Start the dev stack (viewer 5173 + API 8000) in a separate terminal.
make serve

# 2. Run the validation walk against your corpus.
make ci-ui-validation CORPUS=/Users/me/.test_outputs/manual/my-corpus

# Equivalent direct invocation:
cd web/gi-kg-viewer && \
  CORPUS_PATH=/Users/me/.test_outputs/manual/my-corpus \
  node_modules/.bin/playwright test --config playwright.validation.config.ts
```

Screenshots land in `web/gi-kg-viewer/validation-results/v*.png`.

## What it covers

| Scenario | Surface | Expected outcome |
| --- | --- | --- |
| V1 | Library → Open in graph | Episode selected, camera centered, rail shows title |
| V2 | Digest topic pill | Either applies cleanly OR fails with visible error strip (V2 fix from RFC-086) |
| V3 | Search "Show on graph" | Requires real vector index — skip if `index_stats.available=false` |
| V4 | Dashboard topic-cluster chip | Compound parent selected, camera centered |
| V5 | Hot-state Library → Library | Second click supersedes cleanly within 15 s |

The companion file `handoff-matrix-real-corpus.spec.ts` extends Tier-3
to **29 of the 41 `HANDOFF_MATRIX.md` rows** (V1–V5 plus P1.3, P1.6,
P1.10, P1.12, P1.13, P2.2, P2.3, P2.4, P2.5, P2.6, P3.1, P3.2, P3.3,
P4.1, P4.2, P4.3, P5.1, P5.2, P6.1, P6.2, P7.1, P7.2, P8.1, P8.2, P8.3,
P8.4, P8.5, P8.6, P8.7). The remaining 12 rows are either N/A
(P1.4 — affordance disabled), duplicates of covered rows (P6.3 ⊆ P6.2),
or require complex multi-step UI setup that's better suited to Tier-2.

Each scenario asserts the same 6-point contract used by Tier 1 + Tier 2:

- FSM `state === 'ready'` and `lastResult.status === 'applied'` (or
  explicit `failed` for known-bucket cases)
- `cy.nodes(':selected').length === 1` with id matching expected
- Camera zoom in sane range AND node centered in viewport (GH #771 class)
- Subject store reflects the target
- Self-healing invariant: `expected ⊖ actual === ∅`
- Zero console errors

## Test-plumbing limit: synthetic cytoscape events

Tier-3 cannot reliably synthesize a `tap` / `onetap` event that routes
through Cytoscape's renderer + `core.on('tap', handler)` listener.

Tried and found unreliable:

- `cy.emit('tap', {target: cy})` from `page.evaluate` — extraParams
  unpacking doesn't always produce the right `evt.target`
- `page.mouse.click(x, y)` on the canvas DOM — the DOM event reaches the
  `<canvas>` but Cytoscape's tap-detection state machine (mousedown
  timestamp + movement threshold + mouseup) doesn't always trigger on
  Playwright's instant click

**What this means for the matrix:**

- **Tier-3 catches** the *negative* contract for canvas-tap rows (P1.13
  background tap, P3.3 canvas single-tap): a tap MUST NOT fire
  `handoffRequested`. This is the regression class "did someone
  accidentally route a tap through cross-surface navigation?"
- **Tier-1** (`e2e/handoff/cold-start.spec.ts::H1.13`, `repeat-click.spec.ts::H3.3`)
  covers the *positive* contract (selection cleared, `canvasTapped`
  envelope fires) deterministically via mock-driven cy state — no
  real renderer in the loop.
- **Production users** are unaffected: real mice send real DOM events
  with proper timing that Cytoscape's handler accepts.

Don't burn time trying to fix this — it's an intrinsic limit of
synthetic event harnesses against renderer-driven canvases, not a hole
in product coverage.

## When to run

- Before pushing to main (per CLAUDE.md "Final validation before push")
- After any change to `GraphCanvas.vue`, handoff store, or entry-point
  components
- Weekly via scheduled GHA (when wired — see RFC-086 Phase 5 open item)

## Institutional rule (RFC-086)

**Every bug surfaced by Tier 3 must land a Tier 2 matrix row that
reproduces it before the fix PR merges.** Bugs become structurally
non-regressable. Add the reproducing row under
`web/gi-kg-viewer/e2e/handoff-production/` — same 6-point contract via
`assertHandoffApplied`.

## Authoring more validation scenarios

Edit `real-corpus.spec.ts`. The helper pattern:

```typescript
test('V<N> — <surface> (real corpus)', async ({ page }) => {
  const errs = captureConsoleErrors(page)
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await fillCorpusPath(page)

  // ...user interaction...

  const state = await waitForFsmReady(page)
  const report = summariseHandoff('V<N> <label>', state, errs.errors)
  console.log('[validation V<N>]', JSON.stringify(report, null, 2))
  expect(report.fsmState).toBe('ready')
  expect(report.selectedId).toBeTruthy()
  expect(report.zoomOk).toBe(true)
  expect(report.cameraOk).toBe(true)
})
```

The corpus path is taken from `CORPUS_PATH` env var (set by the make
target). If unset, falls back to a hard-coded path in the spec.
