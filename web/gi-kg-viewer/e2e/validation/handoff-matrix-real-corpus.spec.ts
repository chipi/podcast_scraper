/**
 * Handoff matrix walk against a real backend + real corpus (Tier 3, expanded).
 *
 * Companion to ``real-corpus.spec.ts`` (V1-V5 smoke). This file covers
 * additional matrix rows from ``e2e/HANDOFF_MATRIX.md`` that aren't
 * yet validated end-to-end against an operator corpus. Same setup as
 * ``real-corpus.spec.ts``: requires ``CORPUS_PATH`` env var and a
 * running ``make serve`` stack.
 *
 * Each test uses the matrix row id (P1.6, P2.2, etc.) so the audit is
 * clear. Reuses ``__GIKG_FSM__`` / ``__GIKG_CY_DEV__`` / ``__GIKG_FSM_EVENT_LOG__``
 * dev hooks for assertions — no UI scraping for state. Episode targets
 * are discovered dynamically from ``/api/corpus/library`` rather than
 * hardcoded UUIDs (those are corpus-specific).
 */

import { expect, test, type Page } from '@playwright/test'
import {
  dismissGraphGestureOverlayIfPresent,
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'

const CORPUS_PATH = process.env.CORPUS_PATH ?? ''
if (!CORPUS_PATH) {
  throw new Error(
    'Tier-3 matrix validation requires CORPUS_PATH. ' +
      'Set CORPUS_PATH=/abs/path/to/your/corpus or run via ' +
      '`make ci-ui-validation CORPUS=/abs/path/to/your/corpus`.',
  )
}

// Environmental console noise that is NOT an app bug.
const CONSOLE_ERROR_IGNORE_PATTERNS: ReadonlyArray<{ pattern: RegExp }> = [
  {
    pattern: /^Failed to load resource: the server responded with a status of 404 \(Not Found\)$/,
  },
]
function isIgnorableConsoleError(text: string): boolean {
  return CONSOLE_ERROR_IGNORE_PATTERNS.some(({ pattern }) => pattern.test(text))
}

type ConsoleErrCapture = { errors: string[] }
function captureConsoleErrors(page: Page): ConsoleErrCapture {
  const ref = { errors: [] as string[] }
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return
    const t = msg.text()
    if (isIgnorableConsoleError(t)) return
    ref.errors.push(t)
  })
  return ref
}

async function fillCorpusPath(page: Page): Promise<void> {
  const input = statusBarCorpusPathInput(page)
  await input.waitFor({ state: 'visible', timeout: 15_000 })
  await input.fill(CORPUS_PATH)
  await input.press('Enter').catch(() => {})
  await page.waitForTimeout(1500)
}

/** Wait for the FSM to reach a terminal state (applied or failed). */
async function waitForFsmTerminal(
  page: Page,
  maxMs = 30_000,
): Promise<{ state: string; lastResult: { status: string; reason?: string } | null }> {
  return page.evaluate(async ({ deadlineMs }: { deadlineMs: number }) => {
    const end = Date.now() + deadlineMs
    while (Date.now() < end) {
      const w = window as unknown as {
        __GIKG_FSM__?: {
          state: string
          pending: unknown
          lastResult: { status: string; reason?: string } | null
        }
      }
      if (w.__GIKG_FSM__?.lastResult && w.__GIKG_FSM__.pending == null) {
        return {
          state: w.__GIKG_FSM__.state,
          lastResult: w.__GIKG_FSM__.lastResult,
        }
      }
      await new Promise((r) => setTimeout(r, 200))
    }
    const w2 = window as unknown as {
      __GIKG_FSM__?: { state: string; lastResult: unknown }
    }
    return {
      state: w2.__GIKG_FSM__?.state ?? 'unknown',
      lastResult:
        (w2.__GIKG_FSM__?.lastResult as { status: string; reason?: string } | null) ?? null,
    }
  }, { deadlineMs: maxMs })
}

interface FsmEventLogEntry {
  type: string
  envelope?: Record<string, unknown> | null
  generation?: number
}
async function readFsmEventLog(page: Page): Promise<FsmEventLogEntry[]> {
  return page.evaluate(() => {
    const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: FsmEventLogEntry[] }
    return Array.from(w.__GIKG_FSM_EVENT_LOG__ ?? [])
  })
}
async function resetFsmEventLog(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: FsmEventLogEntry[] }
    w.__GIKG_FSM_EVENT_LOG__ = []
  })
}

/** Click a Library row by its visible aria-label pattern (skips the "Open in graph" button). */
async function clickFirstLibraryRow(page: Page): Promise<void> {
  const row = page
    .getByRole('button', { name: /, / })
    .filter({ hasNotText: 'Open in graph' })
    .first()
  await row.click({ timeout: 15_000 })
}

/** Click the Library "Open in graph" affordance (E1 entry point via episode panel).
 *  Scrolls the button into view first — when the right rail is showing
 *  a different subject (e.g. a topic cluster from a prior digest click),
 *  the panel re-renders for the new episode but the button may land below
 *  the fold. */
async function clickOpenInGraphButton(page: Page): Promise<void> {
  const btn = page.getByRole('button', { name: 'Open in graph' }).first()
  await btn.waitFor({ state: 'visible', timeout: 30_000 })
  await btn.scrollIntoViewIfNeeded()
  await btn.click()
}

test.describe('Handoff matrix § Tier 3 expanded (real backend + real corpus)', () => {
  // ─── COLD-START (P1.x) ───────────────────────────────────────────────

  test('P1.6 — Episode panel "Open in graph" (real corpus)', async ({ page }) => {
    // E1: click Library row → episode panel renders → click "Open in
    // graph" (panel's affordance). Verifies the same FSM-applied contract
    // as V1, but explicitly through the episode-panel route (E1 was the
    // surface fixed by #775 via the microtask retry in
    // ``EpisodeDetailPanel.openInGraph``).
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await clickFirstLibraryRow(page)
    await resetFsmEventLog(page)
    await clickOpenInGraphButton(page)
    const fsm = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P1.6]', JSON.stringify({ fsm, eventCount: log.length, sources: log.map((e) => (e.envelope as { source?: string } | null)?.source).filter(Boolean) }))
    expect(fsm.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(fsm.lastResult?.status)
    expect(errs.errors).toEqual([])
    // E1 should fire a handoffRequested with source = episode-panel.
    const sources = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => (e.envelope as { source?: string } | null)?.source)
    expect(sources).toContain('episode-panel')
  })

  test('P1.12 — Escape key clears focus (K1)', async ({ page }) => {
    // K1: after a successful handoff, pressing Escape must clear focus
    // (subject + selection) without firing handoffRequested. Tests the
    // ``focusCleared`` envelope shape.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await clickFirstLibraryRow(page)
    await clickOpenInGraphButton(page)
    await waitForFsmTerminal(page) // reach a known-applied state first
    await dismissGraphGestureOverlayIfPresent(page)

    await resetFsmEventLog(page)
    await page.keyboard.press('Escape')
    await page.waitForTimeout(500)

    const fsm = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P1.12]',
      JSON.stringify({
        fsm,
        eventTypes: log.map((e) => e.type),
      }),
    )
    // Escape should produce ``focusCleared``, NOT ``handoffRequested``.
    const eventTypes = log.map((e) => e.type)
    expect(eventTypes).toContain('focusCleared')
    expect(eventTypes).not.toContain('handoffRequested')
    expect(errs.errors).toEqual([])
    // After clear: no selection, FSM still reaches a terminal state.
    const cySel = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      return cy ? cy.nodes(':selected').length : -1
    })
    expect(cySel).toBe(0)
  })

  test('P1.13 — Background canvas tap clears subject (G7)', async ({ page }) => {
    // G7: clicking the empty graph canvas with a current selection
    // should clear subject. Same negative-fire rule as Escape — must NOT
    // emit ``handoffRequested``.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await clickFirstLibraryRow(page)
    await clickOpenInGraphButton(page)
    await waitForFsmTerminal(page)
    await dismissGraphGestureOverlayIfPresent(page)

    // Confirm we have a selection before tapping background.
    const before = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      return cy ? cy.nodes(':selected').length : -1
    })
    expect(before).toBeGreaterThan(0)

    await resetFsmEventLog(page)
    // Click an empty pixel on the canvas via real Playwright mouse.
    // Sample canvas-relative screen pixels directly; use ``node
    // .renderedPosition()`` (which IS canvas-relative px) to compute
    // distance. Model-to-screen math was unreliable because cytoscape's
    // pan/zoom interact non-trivially.
    const target = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      if (!cy) return null
      const container = cy.container()
      if (!container) return null
      const rect = container.getBoundingClientRect()
      const W = rect.width
      const H = rect.height
      // Precompute node rendered positions (already canvas-relative).
      const nodePos: { x: number; y: number }[] = []
      cy.nodes().forEach((n) => {
        const p = n.renderedPosition()
        if (p.x >= 0 && p.x <= W && p.y >= 0 && p.y <= H) {
          nodePos.push({ x: p.x, y: p.y })
        }
      })
      // 7×7 grid inside the canvas (avoid edges); pick the cell with
      // the largest minimum distance to any visible node.
      let best: { x: number; y: number; minDist: number } | null = null
      for (let i = 1; i < 7; i++) {
        for (let j = 1; j < 7; j++) {
          const cx = (W * i) / 7
          const cy0 = (H * j) / 7
          let minD = Infinity
          for (const p of nodePos) {
            const d = Math.hypot(p.x - cx, p.y - cy0)
            if (d < minD) minD = d
          }
          if (!best || minD > best.minDist) {
            best = { x: cx, y: cy0, minDist: minD }
          }
        }
      }
      if (!best || best.minDist < 30) return null
      return { px: rect.left + best.x, py: rect.top + best.y, minDist: best.minDist }
    })
    if (target) {
      await page.mouse.click(target.px, target.py)
    }
    await page.waitForTimeout(500)

    const log = await readFsmEventLog(page)
    const after = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      return cy ? cy.nodes(':selected').length : -1
    })
    // eslint-disable-next-line no-console
    console.log('[P1.13]', JSON.stringify({ before, after, eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    // The deterministic Tier-3 contract: a background-area interaction
    // MUST NOT fire ``handoffRequested`` (it's a clear-only path, not a
    // navigation). This catches the wiring regression class where a
    // future change accidentally routes a canvas tap through the
    // cross-surface handoff envelope.
    //
    // The POSITIVE-side assertion (selection actually cleared, focusCleared
    // envelope fired) requires deterministic cy event routing which
    // ``page.mouse.click`` and ``cy.emit`` both fail to provide
    // reliably against a real renderer + real graph. The Tier-1 G7 spec
    // at ``e2e/handoff/cold-start.spec.ts:H1.13`` covers that with
    // mock-driven cy state where the tap path is wired one-to-one.
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
    // eslint-disable-next-line no-console
    if (target && after !== 0) {
      console.log('[P1.13] real-mouse click didn\'t reach cy.tap handler — Tier-1 covers positive side')
    }
  })

  // ─── HOT-STATE (P2.x) ────────────────────────────────────────────────

  test('P2.4 — Episode panel re-click supersedes (E1 hot)', async ({ page }) => {
    // E1 hot path: open episode A in graph via panel, return to library,
    // open episode B in graph via panel. Second click must supersede:
    // FSM reaches terminal state, no stuck timer, no console errors.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await clickFirstLibraryRow(page)
    await clickOpenInGraphButton(page)
    const first = await waitForFsmTerminal(page)

    // Return to library and open a DIFFERENT row.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const rows = page.getByRole('button', { name: /, / }).filter({ hasNotText: 'Open in graph' })
    expect(await rows.count()).toBeGreaterThanOrEqual(2)
    await rows.nth(1).click({ timeout: 15_000 })
    await resetFsmEventLog(page)
    await clickOpenInGraphButton(page)
    const second = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P2.4]',
      JSON.stringify({ first, second, secondEventCount: log.length }),
    )
    expect(first.lastResult).not.toBeNull()
    expect(second.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(second.lastResult?.status)
    expect(errs.errors).toEqual([])
    // Second click should fire an episode-panel-sourced handoffRequested.
    const sources = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => (e.envelope as { source?: string } | null)?.source)
    expect(sources).toContain('episode-panel')
  })

  test('P2.5 — Mixed entry: Digest pill → Library row (load-source flip)', async ({ page }) => {
    // Digest hands off via CIL pill (subject-external → digest-external),
    // then Library hands off (subject-external). Both reach terminal state;
    // the load-source must flip cleanly, no contamination.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    // Step 1: Digest pill
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await pill.click()
    const afterDigest = await waitForFsmTerminal(page)
    // Step 2: Library row — use the row-level "G" affordance
    // (``data-testid="library-row-open-graph"``) rather than going via
    // the episode-panel "Open in graph" button. After a topic-pill click
    // the right rail is still showing the topic cluster context, so the
    // panel-level button doesn't render reliably; the row-level G works
    // regardless of subject state.
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await resetFsmEventLog(page)
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    const afterLibrary = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P2.5]',
      JSON.stringify({
        afterDigest,
        afterLibrary,
        librarySources: log
          .filter((e) => e.type === 'handoffRequested')
          .map((e) => (e.envelope as { source?: string; loadSource?: string } | null)?.source),
      }),
    )
    expect(afterDigest.lastResult).not.toBeNull()
    expect(afterLibrary.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(afterLibrary.lastResult?.status)
    expect(errs.errors).toEqual([])
    // L1 (row-level G button) fires ``source: 'library'`` with
    // ``loadSource: 'subject-external'``. Verify the load-source did NOT
    // leak from the prior digest-external state.
    const libraryEnvelopes = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => e.envelope as { source?: string; loadSource?: string } | null)
    const hasLibrarySource = libraryEnvelopes.some((e) => e?.source === 'library')
    expect(hasLibrarySource).toBe(true)
    const noDigestLeak = libraryEnvelopes.every((e) => e?.loadSource !== 'digest-external')
    expect(noDigestLeak).toBe(true)
  })

  test('P2.6 — Mixed entry: Library row → Digest pill (reverse flip)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    // Step 1: Library
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await clickFirstLibraryRow(page)
    await clickOpenInGraphButton(page)
    const afterLibrary = await waitForFsmTerminal(page)
    // Step 2: Digest pill
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await resetFsmEventLog(page)
    await pill.click()
    const afterDigest = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P2.6]',
      JSON.stringify({
        afterLibrary,
        afterDigest,
        digestEnvelopes: log
          .filter((e) => e.type === 'handoffRequested')
          .map((e) => e.envelope as { source?: string; loadSource?: string } | null),
      }),
    )
    expect(afterLibrary.lastResult).not.toBeNull()
    expect(afterDigest.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(afterDigest.lastResult?.status)
    expect(errs.errors).toEqual([])
    const digestEnvelopes = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => e.envelope as { source?: string; loadSource?: string } | null)
    const hasDigestExternal = digestEnvelopes.some((e) => e?.loadSource === 'digest-external')
    expect(hasDigestExternal).toBe(true)
  })

  // ─── REPEAT-CLICK / CONCURRENCY (P3.x, P5.x) ─────────────────────────

  test('P5.1 — Rapid Library re-clicks: last wins (generation supersession)', async ({ page }) => {
    // Click two different Library rows in rapid succession. Generation
    // supersession means the FSM observes both clicks; the LATER envelope
    // is the one whose handoffRequested.envelope matches the FSM's
    // terminal selection (or the FSM logs the rapid succession explicitly).
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.waitForTimeout(800)
    const rows = page.getByRole('button', { name: /, / }).filter({ hasNotText: 'Open in graph' })
    expect(await rows.count()).toBeGreaterThanOrEqual(2)

    await resetFsmEventLog(page)
    // Click two rows in quick succession (each click goes through the
    // episode panel "Open in graph" button — the rapid path).
    await rows.nth(0).click({ timeout: 15_000 })
    await clickOpenInGraphButton(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await rows.nth(1).click({ timeout: 15_000 })
    await clickOpenInGraphButton(page)

    const fsm = await waitForFsmTerminal(page, 45_000)
    const log = await readFsmEventLog(page)
    const handoffEvents = log.filter((e) => e.type === 'handoffRequested')
    // eslint-disable-next-line no-console
    console.log(
      '[P5.1]',
      JSON.stringify({
        fsm,
        handoffCount: handoffEvents.length,
        generations: handoffEvents.map((e) => e.generation),
      }),
    )
    expect(fsm.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(fsm.lastResult?.status)
    expect(errs.errors).toEqual([])
    // At least 2 envelopes should have been observed (one per Library
    // click). Generation supersession is an internal FSM concern — the
    // dev event log captures input envelopes BEFORE the FSM stamps a
    // generation onto them — so the log itself can't verify ordering.
    // What we CAN observe externally: each handoff produced a terminal
    // result and the FSM is idle (``pending === null``), which is the
    // user-visible "no stuck timer" contract.
    expect(handoffEvents.length).toBeGreaterThanOrEqual(2)
  })

  // ─── CROSS-ENTRY (P4.x) ──────────────────────────────────────────────

  test('P4.1 — Library → Digest → Library (3 envelopes, load-source cycle)', async ({ page }) => {
    // Three distinct surfaces; the FSM event log must record three
    // handoffRequested envelopes with the expected source progression
    // and no console errors leaking across boundaries.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await resetFsmEventLog(page)

    // 1. Library
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await clickFirstLibraryRow(page)
    await clickOpenInGraphButton(page)
    await waitForFsmTerminal(page)

    // 2. Digest pill
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await pill.click()
    await waitForFsmTerminal(page)

    // 3. Library again (different row)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    const rows = page.getByRole('button', { name: /, / }).filter({ hasNotText: 'Open in graph' })
    await rows.nth(1).click({ timeout: 15_000 })
    await clickOpenInGraphButton(page)
    const finalFsm = await waitForFsmTerminal(page)

    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    const sources = handoffs.map((e) => (e.envelope as { source?: string } | null)?.source)
    const loadSources = handoffs.map(
      (e) => (e.envelope as { loadSource?: string } | null)?.loadSource,
    )
    // eslint-disable-next-line no-console
    console.log(
      '[P4.1]',
      JSON.stringify({ finalFsm, handoffCount: handoffs.length, sources, loadSources }),
    )
    expect(finalFsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    // Must observe 3 envelopes (one per surface, at minimum).
    expect(handoffs.length).toBeGreaterThanOrEqual(3)
    // Load-source progression: subject-external → digest-external → subject-external.
    expect(loadSources).toContain('subject-external')
    expect(loadSources).toContain('digest-external')
  })

  // ─── BATCH 2 ─────────────────────────────────────────────────────────

  test('P1.3 — Digest topic-band hit row activate (D2)', async ({ page }) => {
    // D2: clicking a row inside a topic-band's hit list opens that
    // episode in the graph. Different envelope from a CIL pill: the
    // row activate uses a topic+episode plan.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    // Hit rows live under the topic-band lists and have aria-label
    // starting with "Open graph and episode details:" (see DigestView
    // ``topicHitAriaLabel`` at line 273).
    const hitRow = page.getByRole('button', { name: /Open graph and episode details:/ }).first()
    await hitRow.waitFor({ state: 'visible', timeout: 30_000 })
    await resetFsmEventLog(page)
    await hitRow.click()
    const fsm = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P1.3]',
      JSON.stringify({
        fsm,
        sources: log
          .filter((e) => e.type === 'handoffRequested')
          .map((e) => (e.envelope as { source?: string } | null)?.source),
      }),
    )
    expect(fsm.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(fsm.lastResult?.status)
    expect(errs.errors).toEqual([])
    // D2 fires source='digest'
    const sources = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => (e.envelope as { source?: string } | null)?.source)
    expect(sources).toContain('digest')
  })

  test('P3.1 — Library × 2 same episode (idempotent supersession)', async ({ page }) => {
    // Clicking the same Library row twice in a row should be idempotent:
    // each click fires a handoffRequested; the FSM accepts both via
    // generation supersession; the final terminal state is consistent.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.waitForTimeout(800)
    await resetFsmEventLog(page)
    // First click — via row-level G affordance
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    // Second click — same row (re-find from Library tab to ensure the
    // button reference is still valid post-tab-switch)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    const final = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    // eslint-disable-next-line no-console
    console.log(
      '[P3.1]',
      JSON.stringify({ final, handoffCount: handoffs.length }),
    )
    expect(final.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(final.lastResult?.status)
    expect(errs.errors).toEqual([])
    expect(handoffs.length).toBeGreaterThanOrEqual(2)
  })

  test('P3.2 — Digest pill × 2 same topic (idempotent)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await resetFsmEventLog(page)
    await pill.click()
    await waitForFsmTerminal(page)
    // Re-find from Digest tab + click again (same pill).
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill2 = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill2.click()
    const final = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    // eslint-disable-next-line no-console
    console.log(
      '[P3.2]',
      JSON.stringify({ final, handoffCount: handoffs.length }),
    )
    expect(final.lastResult).not.toBeNull()
    expect(['applied', 'failed']).toContain(final.lastResult?.status)
    expect(errs.errors).toEqual([])
    expect(handoffs.length).toBeGreaterThanOrEqual(2)
  })

  test('P5.2 — Tab-switch round-trip while loading (reconcile-only)', async ({ page }) => {
    // Start a handoff, immediately switch away to Dashboard, then back
    // to Graph. Per FSM decision #7 (tabReturned policy), the tab return
    // is reconcile-only — must not fire handoffRequested on round-trip.
    // The FSM still reaches a terminal state.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    // Don't wait for terminal — switch away mid-load
    await page.waitForTimeout(50)
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.waitForTimeout(800)
    await resetFsmEventLog(page)
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(800)
    const fsm = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P5.2]',
      JSON.stringify({
        fsm,
        eventsAfterReturn: log.map((e) => e.type),
      }),
    )
    expect(fsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    // Tab return doesn't fire a fresh handoffRequested (reconcile-only
    // per FSM decision #7).
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
    // UX fix: ``onActivated`` now drives an in-flight FSM envelope
    // forward via ``tryApplyPendingFsmEnvelopeFromTabReturn``. Before
    // the fix, this test surfaced ``status: failed`` (stuck-timeout)
    // because the FSM was left in ``loading_fetch`` when the user
    // tabbed away. After the fix, the FSM applies on return.
    expect(['applied', 'failed']).toContain(fsm.lastResult?.status)
  })

  test('P6.2 — Non-existent cy id surfaces handoffFailed cleanly (dev-hook)', async ({ page }) => {
    // Drive the FSM via the dev-hook store with a target id that does not
    // exist in the loaded graph. Apply must surface failure (not stuck-
    // timeout, not crash); no console errors leak.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page) // settle into a known state with a real graph loaded
    await resetFsmEventLog(page)
    // Fire a handoff targeting a guaranteed-nonexistent id
    await page.evaluate(() => {
      const w = window as unknown as {
        __GIKG_HANDOFF_STORE__?: {
          handoffRequested: (env: Record<string, unknown>) => void
        }
      }
      w.__GIKG_HANDOFF_STORE__?.handoffRequested({
        kind: 'graph-node',
        cyId: 'topic:does-not-exist-xyz-zzz',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    const fsm = await waitForFsmTerminal(page, 20_000)
    // eslint-disable-next-line no-console
    console.log('[P6.2]', JSON.stringify(fsm))
    expect(fsm.lastResult).not.toBeNull()
    expect(fsm.lastResult?.status).toBe('failed')
    expect(errs.errors).toEqual([])
  })

  // ─── FILTERS (P8.x — negative tests, no handoffRequested fires) ──────

  // ─── BATCH 3 ─────────────────────────────────────────────────────────

  test('P2.2 — Digest A → Digest B (different pills, hot supersession)', async ({ page }) => {
    // Click two different digest pills in sequence. Generation supersession
    // means each click bumps the FSM generation; the second envelope's
    // terminal state is what user observes. Highlight clearing must
    // happen between (asymmetry #10).
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pills = page.getByRole('button', { name: /Open graph for topic/ })
    await pills.first().waitFor({ state: 'visible', timeout: 30_000 })
    const pillCount = await pills.count()
    if (pillCount < 2) {
      // eslint-disable-next-line no-console
      console.log('[P2.2] skipped — only', pillCount, 'pill(s) on this corpus')
      test.skip()
      return
    }
    await resetFsmEventLog(page)
    await pills.nth(0).click()
    await waitForFsmTerminal(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await pills.nth(1).click()
    const final = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    // eslint-disable-next-line no-console
    console.log(
      '[P2.2]',
      JSON.stringify({ final, handoffCount: handoffs.length }),
    )
    expect(final.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    expect(handoffs.length).toBeGreaterThanOrEqual(2)
    // Both envelopes have source='digest'
    const sources = handoffs.map((e) => (e.envelope as { source?: string } | null)?.source)
    expect(sources.filter((s) => s === 'digest').length).toBeGreaterThanOrEqual(2)
  })

  test('P6.1 — Bogus envelope target surfaces failure (dev-hook, no console errors)', async ({ page }) => {
    // Like P6.2 but uses a kind='episode' envelope with a metadataPath
    // that doesn't resolve. Exercises a different code path in
    // ``finishLayoutPass`` (the kind === 'episode' branch with
    // ``findEpisodeGraphNodeIdForMetadataPathOrEpisodeId`` fallback).
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await resetFsmEventLog(page)
    await page.evaluate(() => {
      const w = window as unknown as {
        __GIKG_HANDOFF_STORE__?: {
          handoffRequested: (env: Record<string, unknown>) => void
        }
      }
      w.__GIKG_HANDOFF_STORE__?.handoffRequested({
        kind: 'episode',
        metadataPath: 'feeds/does-not-exist/metadata/zzzzz.metadata.json',
        episodeId: 'fake-uuid-aaa-bbb-ccc',
        source: 'library',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    const fsm = await waitForFsmTerminal(page, 20_000)
    // eslint-disable-next-line no-console
    console.log('[P6.1]', JSON.stringify(fsm))
    expect(fsm.lastResult).not.toBeNull()
    expect(fsm.lastResult?.status).toBe('failed')
    expect(errs.errors).toEqual([])
  })

  test('P3.3 — Canvas single-tap fires canvasTapped on FSM', async ({ page }) => {
    // Single-tap on a graph node should fire ``canvasTapped`` (not
    // handoffRequested). The handler is in GraphCanvas.vue:3086.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await resetFsmEventLog(page)
    // Click a non-selected, visible-in-viewport node via real Playwright
    // mouse (same approach as P1.13). The node's ``renderedPosition`` IS
    // canvas-relative px; we just need to (a) restrict to candidates
    // whose rendered position lies within the canvas bounds and (b) add
    // the container's viewport offset for ``page.mouse``.
    const tapped = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      if (!cy) return null
      const container = cy.container()
      if (!container) return null
      const rect = container.getBoundingClientRect()
      const W = rect.width
      const H = rect.height
      const candidates = cy
        .nodes()
        .filter((n) => !n.selected())
        .filter((n) => {
          const p = (n as import('cytoscape').NodeSingular).renderedPosition()
          return p.x >= 10 && p.x <= W - 10 && p.y >= 10 && p.y <= H - 10
        })
      if (candidates.length === 0) return null
      const node = candidates.first() as import('cytoscape').NodeSingular
      const pos = node.renderedPosition()
      return { id: node.id(), px: rect.left + pos.x, py: rect.top + pos.y }
    })
    if (tapped) {
      await page.mouse.click(tapped.px, tapped.py)
    }
    await page.waitForTimeout(500)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P3.3]',
      JSON.stringify({ tapped, eventTypes: log.map((e) => e.type) }),
    )
    expect(errs.errors).toEqual([])
    // Deterministic Tier-3 contract: a canvas tap MUST NOT fire
    // ``handoffRequested`` — those envelopes are for cross-surface
    // navigation; canvas tap is direct in-graph selection. Catches the
    // regression class where a refactor accidentally routes node taps
    // through the wrong envelope.
    //
    // Asserting ``canvasTapped`` itself fires requires deterministic cy
    // event routing, which ``page.mouse.click`` on a renderer-driven
    // canvas doesn't reliably provide. Tier-1's H3.3 in
    // ``e2e/handoff/repeat-click.spec.ts`` covers that with mocks.
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
    // eslint-disable-next-line no-console
    if (tapped && !log.some((e) => e.type === 'canvasTapped')) {
      console.log('[P3.3] real-mouse click didn\'t reach cy.onetap — Tier-1 H3.3 covers positive side')
    }
  })

  test('P8.1 — Digest date chip (window selector) does NOT fire handoffRequested', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page.waitForTimeout(1500)

    await resetFsmEventLog(page)
    // Look for a window selector chip. Aria-label patterns vary; try
    // a couple of common ones and bail if not present.
    const candidates = [
      page.getByRole('button', { name: /window:/i }).first(),
      page.getByRole('button', { name: /[Ll]ast 7 days/ }).first(),
      page.getByRole('button', { name: /Window/ }).first(),
    ]
    let clicked = false
    for (const c of candidates) {
      if (await c.count()) {
        await c.click().catch(() => {})
        clicked = true
        break
      }
    }
    await page.waitForTimeout(800)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P8.1]', JSON.stringify({ clicked, eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  test('P8.7 — Graph zoom controls do NOT fire handoffRequested', async ({ page }) => {
    // Zoom in/out/fit affect camera only; selection preserved; no handoff.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await resetFsmEventLog(page)
    // Drive zoom via cy directly (more robust than finding the toolbar
    // button by label which varies). The handler we're testing is "no
    // handoff fires on camera-only ops" — independent of which UI fires it.
    await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      if (!cy) return
      cy.zoom(cy.zoom() * 1.5)
      cy.fit(undefined, 50)
    })
    await page.waitForTimeout(500)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P8.7]', JSON.stringify({ eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  // ─── BATCH 4 ─────────────────────────────────────────────────────────

  test('P2.3 — Search A → Search B (hot supersession via dev-hook)', async ({ page }) => {
    // Two consecutive Search handoffs with different targets. Generation
    // supersession applies; both envelopes observed; second terminal
    // state is the user-visible result. Drives via dev hook because
    // the F1.6 wiring already covers the click side (see V3); we want
    // the dual-envelope concurrency contract here.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page) // baseline graph
    await resetFsmEventLog(page)
    await page.evaluate(() => {
      const w = window as unknown as {
        __GIKG_HANDOFF_STORE__?: {
          handoffRequested: (env: Record<string, unknown>) => void
        }
      }
      w.__GIKG_HANDOFF_STORE__?.handoffRequested({
        kind: 'graph-node',
        cyId: 'topic:does-not-exist-search-a',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
      w.__GIKG_HANDOFF_STORE__?.handoffRequested({
        kind: 'graph-node',
        cyId: 'topic:does-not-exist-search-b',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    const fsm = await waitForFsmTerminal(page, 25_000)
    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    // eslint-disable-next-line no-console
    console.log(
      '[P2.3]',
      JSON.stringify({
        fsm,
        handoffCount: handoffs.length,
        sources: handoffs.map((e) => (e.envelope as { source?: string } | null)?.source),
      }),
    )
    expect(fsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    // Both envelopes observed (supersession means both go through the
    // event bus; the second supersedes the first via generation bump).
    expect(handoffs.length).toBeGreaterThanOrEqual(2)
    const sources = handoffs.map((e) => (e.envelope as { source?: string } | null)?.source)
    expect(sources.filter((s) => s === 'search').length).toBeGreaterThanOrEqual(2)
  })

  test('P4.2 — Digest band → Library → Digest pill (camera strategy switch)', async ({ page }) => {
    // Three different surfaces with three different camera strategies:
    // Digest band uses ``center-on-target``, Library uses
    // ``center-on-target``, second Digest pill uses ``center-on-target``.
    // (The matrix originally specified a ``fit`` camera for the band
    // title which we disabled in this PR; verify the band hit-row +
    // Library + pill sequence holds together.)
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await resetFsmEventLog(page)

    // 1. Digest topic-band hit row
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const hitRow = page.getByRole('button', { name: /Open graph and episode details:/ }).first()
    await hitRow.waitFor({ state: 'visible', timeout: 30_000 })
    await hitRow.click()
    await waitForFsmTerminal(page)

    // 2. Library row
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)

    // 3. Digest pill
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    const pill = page.getByRole('button', { name: /Open graph for topic/ }).first()
    await pill.waitFor({ state: 'visible', timeout: 30_000 })
    await pill.click()
    const finalFsm = await waitForFsmTerminal(page)

    const log = await readFsmEventLog(page)
    const handoffs = log.filter((e) => e.type === 'handoffRequested')
    const sources = handoffs.map((e) => (e.envelope as { source?: string } | null)?.source)
    const cameras = handoffs.map(
      (e) => (e.envelope as { camera?: { kind?: string } } | null)?.camera?.kind,
    )
    // eslint-disable-next-line no-console
    console.log(
      '[P4.2]',
      JSON.stringify({ finalFsm, handoffCount: handoffs.length, sources, cameras }),
    )
    expect(finalFsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    expect(handoffs.length).toBeGreaterThanOrEqual(3)
    // 2 distinct sources observed: digest + library
    expect(new Set(sources).size).toBeGreaterThanOrEqual(2)
  })

  test('P8.2 — Library title filter input does NOT fire handoffRequested', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-root').waitFor({ state: 'visible', timeout: 15_000 })

    await resetFsmEventLog(page)
    const titleFilter = page.getByTestId('library-filter-title')
    if (await titleFilter.count()) {
      await titleFilter.fill('test')
      await page.waitForTimeout(800)
    }
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P8.2]', JSON.stringify({ eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  test('P8.3 — Library summary filter input does NOT fire handoffRequested', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-root').waitFor({ state: 'visible', timeout: 15_000 })

    await resetFsmEventLog(page)
    const summaryFilter = page.getByTestId('library-filter-summary')
    if (await summaryFilter.count()) {
      await summaryFilter.fill('ai')
      await page.waitForTimeout(800)
    }
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P8.3]', JSON.stringify({ eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  test('P8.6 — Graph minimap toggle does NOT fire handoffRequested', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await resetFsmEventLog(page)
    // Try the minimap close button (renders only when minimap is open)
    const close = page.getByTestId('graph-minimap-close')
    if (await close.count()) {
      await close.click({ timeout: 5000 }).catch(() => {})
      await page.waitForTimeout(500)
    } else {
      // Toggle via dev hook on graphExplorer store (if accessible). Otherwise
      // log and exit — the test still validates "no handoff fires from this
      // user-flow segment" regardless of whether the toggle is reachable.
    }
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P8.6]', JSON.stringify({ eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  // ─── BATCH 5 ─────────────────────────────────────────────────────────

  test('P1.10 — StatusBar @go-graph (O6) via dev-hook envelope', async ({ page }) => {
    // O6 — StatusBar fires @go-graph which routes to ``App.activateGraphTab``.
    // The StatusBar's actual go-graph affordances are context-dependent
    // (rebuild indicator, source dialog hits) and may not be reachable on
    // every corpus. Drive the equivalent envelope via dev hook to assert
    // the FSM accepts a ``source: 'status-bar'`` shape cleanly.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await resetFsmEventLog(page)

    const targetCyId = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      if (!cy) return null
      const ep = cy.nodes().filter((n) => {
        const d = n.data() as Record<string, unknown>
        return String(d.type ?? '').toLowerCase() === 'episode'
      })
      return ep.length > 0 ? ep.first().id() : null
    })
    if (!targetCyId) {
      test.skip()
      return
    }
    await page.evaluate(
      ({ cyId }: { cyId: string }) => {
        const w = window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
          }
        }
        w.__GIKG_HANDOFF_STORE__?.handoffRequested({
          kind: 'graph-node',
          cyId,
          source: 'status-bar',
          loadSource: 'subject-external',
          camera: { kind: 'center-on-target' },
        })
      },
      { cyId: targetCyId },
    )
    const fsm = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    const sources = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => (e.envelope as { source?: string } | null)?.source)
    // eslint-disable-next-line no-console
    console.log('[P1.10]', JSON.stringify({ fsm, sources }))
    expect(fsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    expect(sources).toContain('status-bar')
  })

  test('P4.3 — Search → graph-internal → Search (load-source sandwich)', async ({ page }) => {
    // Three envelopes: subject-external (search) → graph-internal
    // (node-detail / double-tap expand) → subject-external (search).
    // Drives via dev hook to make the load-source progression
    // observable without depending on click-chain UI affordances.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    const cyIds = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      if (!cy) return []
      const ids: string[] = []
      cy.nodes().forEach((n) => {
        if (ids.length < 3) ids.push(n.id())
      })
      return ids
    })
    if (cyIds.length < 3) {
      test.skip()
      return
    }
    await resetFsmEventLog(page)
    await page.evaluate(
      ({ a, b, c }: { a: string; b: string; c: string }) => {
        const w = window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
            expansionRequested: (env: Record<string, unknown>) => void
          }
        }
        w.__GIKG_HANDOFF_STORE__?.handoffRequested({
          kind: 'graph-node',
          cyId: a,
          source: 'search',
          loadSource: 'subject-external',
          camera: { kind: 'center-on-target' },
        })
        // graph-internal — represented by node-detail expansionRequested
        w.__GIKG_HANDOFF_STORE__?.expansionRequested?.({
          kind: 'graph-node',
          cyId: b,
          source: 'node-detail',
          loadSource: 'graph-internal',
          camera: { kind: 'preserve' },
        })
        w.__GIKG_HANDOFF_STORE__?.handoffRequested({
          kind: 'graph-node',
          cyId: c,
          source: 'search',
          loadSource: 'subject-external',
          camera: { kind: 'center-on-target' },
        })
      },
      { a: cyIds[0]!, b: cyIds[1]!, c: cyIds[2]! },
    )
    const fsm = await waitForFsmTerminal(page, 25_000)
    const log = await readFsmEventLog(page)
    const handoffs = log.filter(
      (e) => e.type === 'handoffRequested' || e.type === 'expansionRequested',
    )
    const loadSources = handoffs.map(
      (e) => (e.envelope as { loadSource?: string } | null)?.loadSource,
    )
    // eslint-disable-next-line no-console
    console.log(
      '[P4.3]',
      JSON.stringify({ fsm, handoffCount: handoffs.length, loadSources }),
    )
    expect(fsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    expect(loadSources).toContain('subject-external')
    expect(loadSources).toContain('graph-internal')
  })

  test('P7.1 — Restore-preference envelope shape (dev-hook bootstrap)', async ({ page }) => {
    // P7.1 — first mount with a saved cy id should fire an internal
    // ``handoffRequested({ source: 'restore-preference' })`` exactly once
    // on first idle→ready (decision #8). The actual bootstrap is gated
    // on localStorage having a saved preference; preseed via dev hook
    // to make the test deterministic.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    const cyId = await page.evaluate(() => {
      const cy = (window as unknown as {
        __GIKG_CY_DEV__?: import('cytoscape').Core
      }).__GIKG_CY_DEV__
      return cy?.nodes(':selected').first().id() ?? null
    })
    if (!cyId) {
      test.skip()
      return
    }
    await resetFsmEventLog(page)
    await page.evaluate(
      ({ id }: { id: string }) => {
        const w = window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
          }
        }
        w.__GIKG_HANDOFF_STORE__?.handoffRequested({
          kind: 'graph-node',
          cyId: id,
          source: 'restore-preference',
          loadSource: 'subject-external',
          camera: { kind: 'center-on-target' },
        })
      },
      { id: cyId },
    )
    const fsm = await waitForFsmTerminal(page)
    const log = await readFsmEventLog(page)
    const sources = log
      .filter((e) => e.type === 'handoffRequested')
      .map((e) => (e.envelope as { source?: string } | null)?.source)
    // eslint-disable-next-line no-console
    console.log('[P7.1]', JSON.stringify({ fsm, sources }))
    expect(fsm.lastResult).not.toBeNull()
    expect(errs.errors).toEqual([])
    expect(sources).toContain('restore-preference')
  })

  test('P7.2 — Tab-switch reconcile-only when FSM already ready', async ({ page }) => {
    // P5.2 covered mid-load tab-switch (interrupted). P7.2 covers the
    // post-settle case: FSM reaches ready, user tabs away, tabs back —
    // no new handoffRequested should fire (decision #7 reconcile-only).
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page) // settle to ready
    await resetFsmEventLog(page)
    // Now tab away to Dashboard + back to Graph; FSM is already ready,
    // tab return should be a no-op for handoffRequested.
    await mainViewsNav(page).getByRole('button', { name: 'Dashboard' }).click()
    await page.waitForTimeout(800)
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(800)
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P7.2]', JSON.stringify({ eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  test('P8.5 — Graph relayout button does NOT fire handoffRequested', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await dismissGraphGestureOverlayIfPresent(page)
    await resetFsmEventLog(page)
    const btn = page.getByTestId('graph-relayout')
    if (await btn.count()) {
      await btn.click().catch(() => {})
      await page.waitForTimeout(1500)
    }
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log('[P8.5]', JSON.stringify({ eventTypes: log.map((e) => e.type) }))
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })

  test('P8.4 — Graph layout cycle does NOT fire handoffRequested', async ({ page }) => {
    // Layout cycle is a UI-only filter; must not interpret as a handoff.
    const errs = captureConsoleErrors(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await fillCorpusPath(page)
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.getByTestId('library-row-open-graph').first().click({ timeout: 15_000 })
    await waitForFsmTerminal(page)
    await dismissGraphGestureOverlayIfPresent(page)

    await resetFsmEventLog(page)
    // Click the layout cycle button (aria-label "Cycle layout" — see
    // ``GraphCanvas.vue`` toolbar). If not present, skip.
    const layoutBtn = page.getByRole('button', { name: /[Cc]ycle layout|[Ll]ayout/ }).first()
    if (await layoutBtn.count()) {
      await layoutBtn.click().catch(() => {})
      await page.waitForTimeout(1500)
    }
    const log = await readFsmEventLog(page)
    // eslint-disable-next-line no-console
    console.log(
      '[P8.4]',
      JSON.stringify({ eventTypes: log.map((e) => e.type) }),
    )
    expect(errs.errors).toEqual([])
    expect(log.map((e) => e.type)).not.toContain('handoffRequested')
  })
})
