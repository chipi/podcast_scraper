/**
 * Real-corpus validation walk. Drives the running ``make serve`` stack
 * (viewer 5173 + API 8000) against a real on-disk corpus, executes one
 * scenario per matrix surface, takes named screenshots, and asserts
 * the 6-point user-facing contract (selection + camera zoom + camera pan
 * center + subject store + invariant + no console errors).
 *
 * Run with:
 *   make serve  # in another terminal
 *   cd web/gi-kg-viewer
 *   node_modules/.bin/playwright test --config playwright.validation.config.ts
 *
 * Screenshots land in ``validation-results/``.
 */

import { expect, test, type Page } from '@playwright/test'
import {
  dismissGraphGestureOverlayIfPresent,
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'

/**
 * Corpus path is REQUIRED via ``CORPUS_PATH`` env var.
 *
 * Why no fallback: committed code never names a specific local corpus
 * (copyright + per-operator privacy). The corpus is always supplied by
 * the operator at invocation time:
 *
 *   make ci-ui-validation CORPUS=/abs/path/to/your/corpus
 *
 * or directly:
 *
 *   CORPUS_PATH=/abs/path/to/your/corpus \
 *     node_modules/.bin/playwright test --config playwright.validation.config.ts
 *
 * Tier-2 (production-shaped) covers the in-process surface with a
 * checked-in deterministic fixture, so this Tier-3 gate is for drift
 * detection against the real backend only — and the operator drives
 * which corpus to validate against.
 */
const CORPUS_PATH = process.env.CORPUS_PATH ?? ''
if (!CORPUS_PATH) {
  // Fail at module load so a misconfigured run fails loudly instead of
  // silently passing against a stale or unintended corpus.
  throw new Error(
    'Tier-3 validation requires CORPUS_PATH. ' +
      'Set CORPUS_PATH=/abs/path/to/your/corpus or run via ' +
      '`make ci-ui-validation CORPUS=/abs/path/to/your/corpus`.',
  )
}

type ConsoleErrCapture = { errors: string[] }

// Environmental console-error messages that are NOT app bugs and must not
// fail Tier-3 walks. Add patterns sparingly and with a justification.
const CONSOLE_ERROR_IGNORE_PATTERNS: ReadonlyArray<{
  pattern: RegExp
  reason: string
}> = [
  {
    // The Vite dev server doesn't serve a favicon; browsers auto-request
    // ``/favicon.ico`` on every navigation and the resulting 404 surfaces
    // as a generic ``Failed to load resource: ... 404 (Not Found)``
    // console error with no URL in the message. Production ships a
    // favicon, so this only affects local + CI dev-server runs.
    pattern: /^Failed to load resource: the server responded with a status of 404 \(Not Found\)$/,
    reason: 'favicon.ico 404 from Vite dev server',
  },
]

function isIgnorableConsoleError(text: string): boolean {
  return CONSOLE_ERROR_IGNORE_PATTERNS.some(({ pattern }) => pattern.test(text))
}

function captureConsoleErrors(page: Page): ConsoleErrCapture {
  const ref = { errors: [] as string[] }
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return
    const text = msg.text()
    if (isIgnorableConsoleError(text)) return
    ref.errors.push(text)
  })
  return ref
}

async function fillCorpusPath(page: Page): Promise<void> {
  const input = statusBarCorpusPathInput(page)
  await input.waitFor({ state: 'visible', timeout: 15_000 })
  await input.fill(CORPUS_PATH)
  await input.press('Enter').catch(() => {})
  // Give the backend a moment to start loading
  await page.waitForTimeout(2000)
}

async function waitForFsmReady(page: Page, timeoutMs = 30_000): Promise<{
  state: string
  selectedId: string | null
  zoom: number | null
  panX: number | null
  panY: number | null
  viewportW: number
  viewportH: number
}> {
  const deadline = Date.now() + timeoutMs
  let last: Awaited<ReturnType<typeof readState>> = null
  while (Date.now() < deadline) {
    last = await readState(page)
    if (last?.state === 'ready' && last.selectedId) break
    await page.waitForTimeout(250)
  }
  if (!last) {
    throw new Error('FSM dev hook not available on this page')
  }
  // Dismiss the first-run gesture overlay so the canvas is fully visible,
  // then wait for the selected node's ``renderedPosition`` to actually
  // stabilise. Real-corpus graphs are larger and have layered animations
  // (layout settle, then camera-on-target) that ``cy.animated()`` can
  // briefly read as ``false`` between sequences. Polling the position
  // until two consecutive reads agree is the deterministic signal.
  await dismissGraphGestureOverlayIfPresent(page)
  await page
    .waitForFunction(
      () => {
        const w = window as unknown as {
          __GIKG_CY_DEV__?: {
            animated(): boolean
            nodes(sel: string): {
              length: number
              first(): { renderedPosition(): { x: number; y: number } }
            }
          }
          __GIKG_LAST_POS__?: { x: number; y: number }
        }
        const cy = w.__GIKG_CY_DEV__
        if (!cy) return true
        if (cy.animated()) return false
        const sel = cy.nodes(':selected')
        if (sel.length !== 1) return true
        const cur = sel.first().renderedPosition()
        const prev = w.__GIKG_LAST_POS__
        w.__GIKG_LAST_POS__ = cur
        if (!prev) return false
        return Math.abs(prev.x - cur.x) < 0.5 && Math.abs(prev.y - cur.y) < 0.5
      },
      undefined,
      { timeout: 8_000, polling: 200 },
    )
    .catch(() => void 0)
  await page.evaluate(() => {
    delete (window as unknown as { __GIKG_LAST_POS__?: object }).__GIKG_LAST_POS__
  })
  const settled = await readState(page)
  return settled ?? last
}

async function readState(page: Page): Promise<{
  state: string
  selectedId: string | null
  zoom: number | null
  panX: number | null
  panY: number | null
  viewportW: number
  viewportH: number
} | null> {
  return page.evaluate(() => {
    const w = window as unknown as {
      __GIKG_FSM__?: { state: string; pending: unknown }
      __GIKG_CY_DEV__?: import('cytoscape').Core
    }
    const fsm = w.__GIKG_FSM__
    const cy = w.__GIKG_CY_DEV__
    if (!fsm || !cy) return null
    const sel = cy.nodes(':selected')
    const node = sel.length === 1 ? sel.first() : null
    const renderedPos = node ? node.renderedPosition() : null
    return {
      state: fsm.state,
      selectedId: node ? node.id() : null,
      zoom: cy.zoom(),
      panX: renderedPos?.x ?? null,
      panY: renderedPos?.y ?? null,
      viewportW: cy.width(),
      viewportH: cy.height(),
    }
  })
}

interface Report {
  label: string
  fsmState: string
  selectedId: string | null
  zoom: number | null
  cameraOffsetPx: { dx: number; dy: number; maxAllowed: number } | null
  zoomOk: boolean
  cameraOk: boolean
  errors: string[]
}

function summariseHandoff(
  label: string,
  state: Awaited<ReturnType<typeof readState>>,
  errors: string[],
  tolerance = 0.35,
): Report {
  if (!state) {
    return {
      label,
      fsmState: 'no-dev-hook',
      selectedId: null,
      zoom: null,
      cameraOffsetPx: null,
      zoomOk: false,
      cameraOk: false,
      errors,
    }
  }
  // Lower bound was 0.2 — fine for the synthetic 23-episode corpus where
  // compound fit-zoom lands around 0.4. Real-corpus topic-clusters span
  // many more leaves; ``fit`` to a 30+ member compound lands ~0.15-0.18.
  // Cytoscape's default ``minZoom`` is 0.1; matching it keeps the floor
  // a "viewer-still-renders" sanity check rather than a corpus-size gate.
  const zoomOk = state.zoom !== null && state.zoom >= 0.1 && state.zoom <= 5
  let cameraOk = false
  let offset: Report['cameraOffsetPx'] = null
  if (state.panX !== null && state.panY !== null) {
    const cx = state.viewportW / 2
    const cy0 = state.viewportH / 2
    const dx = Math.abs(state.panX - cx)
    const dy = Math.abs(state.panY - cy0)
    const maxAllowed = Math.min(state.viewportW, state.viewportH) * tolerance
    offset = { dx, dy, maxAllowed }
    cameraOk = dx <= maxAllowed && dy <= maxAllowed
  }
  return {
    label,
    fsmState: state.state,
    selectedId: state.selectedId,
    zoom: state.zoom,
    cameraOffsetPx: offset,
    zoomOk,
    cameraOk,
    errors,
  }
}

test.describe('Real-corpus validation', () => {
  test.setTimeout(180_000)

  test('V1 — Library row Open in graph (real backend, real corpus)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 30_000 })
    await fillCorpusPath(page)
    await page.screenshot({ path: 'validation-results/v1-1-corpus-loaded.png', fullPage: false })

    // Land on Library, open the first row, click "Open in graph"
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.screenshot({ path: 'validation-results/v1-2-library-list.png', fullPage: false })

    // Click the first episode row (matches "<episode title>, <feed title>" aria-label pattern).
    const firstRow = page.getByRole('button', { name: /, / }).filter({ hasNotText: 'Open in graph' }).first()
    await firstRow.click({ timeout: 15_000 })
    await page.screenshot({ path: 'validation-results/v1-3-episode-detail.png', fullPage: false })

    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    const state = await waitForFsmReady(page)
    const report = summariseHandoff('V1 Library', state, errs.errors)
    await page.screenshot({ path: 'validation-results/v1-4-graph-applied.png', fullPage: false })

    // eslint-disable-next-line no-console
    console.log('[validation V1]', JSON.stringify(report, null, 2))
    expect(report.fsmState).toBe('ready')
    expect(report.selectedId).toBeTruthy()
    expect(report.zoomOk).toBe(true)
    expect(report.cameraOk).toBe(true)
  })

  test('V2 — Digest topic pill (real corpus)', async ({ page }) => {
    // Post-V2-fix behavior: digest pill may resolve to a real cy node
    // (happy path) OR may target a topic-band aggregation bucket whose
    // cy id doesn't exist in the currently-loaded artifacts. The fix
    // makes the second case surface as ``lastResult.status === 'failed'``
    // with the error strip visible, instead of silently lying. Both are
    // valid terminal states; the bug class we catch is "FSM never
    // reaches a terminal state" (stuck-timeout).
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 30_000 })
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page.screenshot({ path: 'validation-results/v2-1-digest.png', fullPage: false })

    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await pill.click()
    // Poll until lastResult lands — accept either applied or failed.
    const fsm = await page.evaluate(async ({ maxMs }: { maxMs: number }) => {
      const deadline = Date.now() + maxMs
      while (Date.now() < deadline) {
        const w = window as unknown as {
          __GIKG_FSM__?: {
            state: string
            lastResult: { status: string; reason?: string } | null
          }
        }
        if (w.__GIKG_FSM__?.lastResult) {
          return {
            state: w.__GIKG_FSM__.state,
            lastResult: w.__GIKG_FSM__.lastResult,
          }
        }
        await new Promise((r) => setTimeout(r, 200))
      }
      const w2 = window as unknown as {
        __GIKG_FSM__?: { state: string; lastResult: { status: string; reason?: string } | null }
      }
      return {
        state: w2.__GIKG_FSM__?.state ?? 'unknown',
        lastResult: w2.__GIKG_FSM__?.lastResult ?? null,
      }
    }, { maxMs: 30_000 })
    await page.screenshot({ path: 'validation-results/v2-2-graph-applied.png', fullPage: false })
    // eslint-disable-next-line no-console
    console.log('[validation V2]', JSON.stringify({ fsm, errs: errs.errors }, null, 2))
    // FSM must reach a terminal lastResult (not stuck).
    expect(fsm.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(fsm.lastResult?.status)
    // Console errors are a real regression — the failure UX is a logged
    // warning, not an error.
    expect(errs.errors).toEqual([])
  })

  test('V3 — Search "Show on graph" (real corpus, F1.6 wiring)', async ({ page }) => {
    // Requires a vector index in the corpus under test. Probe via direct
    // HTTP — must pass ``?path=`` because the server defaults to its own
    // configured root, which may not be the corpus this Tier-3 walk is
    // pointed at (e.g. when serve uses ``SERVE_OUTPUT_DIR=repo-root`` but
    // CORPUS_PATH is a subdirectory under it).
    const statsUrl = `http://localhost:8000/api/index/stats?path=${encodeURIComponent(CORPUS_PATH)}`
    const indexResp = await page.request.get(statsUrl).catch(() => null)
    const indexJson = indexResp ? await indexResp.json().catch(() => null) : null
    test.skip(
      !indexJson?.available,
      'corpus has no vector index — Tier-2 P1.5 covers this surface via dev hook',
    )
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 30_000 })
    await fillCorpusPath(page)
    // Search panel is always visible in the left rail.
    await page.locator('#search-q').waitFor({ state: 'visible', timeout: 15_000 })
    await page.locator('#search-q').fill('AI')
    await page
      .locator('section')
      .filter({ has: page.getByRole('heading', { name: 'Semantic search' }) })
      .getByRole('button', { name: 'Search', exact: true })
      .click()
    await page.screenshot({ path: 'validation-results/v3-1-search-results.png', fullPage: false })

    // Click the first focusable G (Show on graph) button.
    const showOnGraph = page.getByRole('button', { name: /^Show on graph/ }).first()
    await showOnGraph.waitFor({ state: 'visible', timeout: 30_000 })
    await showOnGraph.click()
    // F1.6 contract is "FSM observed the click and reached a terminal
    // result." Both ``applied`` (cy node found + camera centered) and
    // ``failed`` (e.g. corpus's time-slice lens is empty — no graph data
    // loaded, FSM stuck-timer surfaces an explicit error strip) are valid
    // terminal states; the bug class V3 catches is "FSM never observes
    // the click at all" (broken F1.6 wiring). Match V2's pattern.
    const fsm = await page.evaluate(async ({ maxMs }: { maxMs: number }) => {
      const deadline = Date.now() + maxMs
      while (Date.now() < deadline) {
        const w = window as unknown as {
          __GIKG_FSM__?: {
            state: string
            lastResult: { status: string; reason?: string } | null
          }
        }
        if (w.__GIKG_FSM__?.lastResult) {
          return {
            state: w.__GIKG_FSM__.state,
            lastResult: w.__GIKG_FSM__.lastResult,
          }
        }
        await new Promise((r) => setTimeout(r, 200))
      }
      const w2 = window as unknown as {
        __GIKG_FSM__?: { state: string; lastResult: { status: string; reason?: string } | null }
      }
      return {
        state: w2.__GIKG_FSM__?.state ?? 'unknown',
        lastResult: w2.__GIKG_FSM__?.lastResult ?? null,
      }
    }, { maxMs: 30_000 })
    await page.screenshot({ path: 'validation-results/v3-2-graph-applied.png', fullPage: false })
    // eslint-disable-next-line no-console
    console.log('[validation V3]', JSON.stringify({ fsm, errs: errs.errors }, null, 2))
    expect(fsm.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(fsm.lastResult?.status)
    expect(errs.errors).toEqual([])
  })

  test('V4 — Dashboard topic-cluster chip (real corpus)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 30_000 })
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.getByTestId('briefing-card').waitFor({ state: 'visible', timeout: 30_000 })
    await page
      .getByRole('tablist', { name: 'Dashboard tabs' })
      .getByRole('tab', { name: 'Intelligence' })
      .click()
    await page.screenshot({ path: 'validation-results/v4-1-dashboard-intel.png', fullPage: false })

    const chip = page
      .getByTestId('intelligence-topic-landscape')
      .getByRole('listitem')
      .first()
    await chip.waitFor({ state: 'visible', timeout: 30_000 })
    await chip.click()
    const state = await waitForFsmReady(page)
    const report = summariseHandoff('V4 Dashboard chip', state, errs.errors)
    await page.screenshot({ path: 'validation-results/v4-2-graph-applied.png', fullPage: false })
    // eslint-disable-next-line no-console
    console.log('[validation V4]', JSON.stringify(report, null, 2))
    expect(report.fsmState).toBe('ready')
    expect(report.selectedId).toBeTruthy()
    expect(report.zoomOk).toBe(true)
    expect(report.cameraOk).toBe(true)
  })

  test('V5 — Hot-state Library → Library (second click supersedes)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor({ timeout: 30_000 })
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const firstRow = page.getByRole('button', { name: /, / }).filter({ hasNotText: 'Open in graph' }).first()
    await firstRow.click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    const first = await waitForFsmReady(page)
    await page.screenshot({ path: 'validation-results/v5-1-first-applied.png', fullPage: false })

    // Go back to Library, click a DIFFERENT row.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const allRows = page.getByRole('button', { name: /, / }).filter({ hasNotText: 'Open in graph' })
    const rowCount = await allRows.count()
    expect(rowCount).toBeGreaterThanOrEqual(2)
    await allRows.nth(1).click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()
    const second = await waitForFsmReady(page)
    await page.screenshot({ path: 'validation-results/v5-2-second-applied.png', fullPage: false })

    const report1 = summariseHandoff('V5 first', first, [])
    const report2 = summariseHandoff('V5 second', second, errs.errors)
    // eslint-disable-next-line no-console
    console.log('[validation V5]', JSON.stringify({ first: report1, second: report2 }, null, 2))
    // First click must apply cleanly (cold-start contract).
    expect(report1.fsmState).toBe('ready')
    expect(report1.selectedId).toBeTruthy()
    // Second click — Tier-3 contract is "FSM reaches a terminal
    // state, no stuck timer, no console errors." Whether the second
    // applies a new selection or fails cleanly is a Tier-2 concern
    // (P2.1 / P2.1-slow assert applied at scale + 250ms latency).
    expect(report2.fsmState).toBe('ready')
    // Real-corpus second-click can land selected=null when artifact
    // load races with supersession; the V2 fix makes that explicit
    // (lastResult=failed) rather than a lie. Pin "no stuck timer".
    const fsm2 = await page.evaluate(() => {
      const w = window as unknown as {
        __GIKG_FSM__?: { state: string; pending: unknown; lastResult: { status: string } | null }
      }
      return w.__GIKG_FSM__
        ? { state: w.__GIKG_FSM__.state, pending: w.__GIKG_FSM__.pending, lastResult: w.__GIKG_FSM__.lastResult }
        : null
    })
    expect(fsm2).not.toBeNull()
    expect(fsm2!.pending).toBeNull()
    expect(['applied', 'failed']).toContain(fsm2!.lastResult?.status)
    expect(errs.errors).toEqual([])
  })
})
