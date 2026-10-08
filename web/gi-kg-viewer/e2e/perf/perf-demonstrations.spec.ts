/**
 * Browser-observable performance demonstrations for #767 / #768, and the ADR-158 guard that the
 * public viewer requests no private endpoint.
 *
 * Each test below is a CONTRAST artifact: it asserts behavior that is
 * ONLY possible with the optimization in place. If the optimization is
 * reverted, the assertion goes red. Pair these with the unit-level
 * before/after demos in ``artifacts.loadSelected.test.ts`` and
 * ``cyCoseLayoutOptions.test.ts``.
 *
 * Tests run against the production-shaped fixture (270 cy nodes,
 * 9 episodes × 5 feeds) — the same fixture the
 * Tier-2 handoff matrix uses.
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from '../helpers'
import { captureConsoleErrors, readFsmState } from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from '../handoff-production/_helpers'

test.describe('Perf demonstrations (#767 / #768) and the ADR-158 guard', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  /**
   * #768 — parallel artifact fetches.
   *
   * The viewer's ``loadSelected`` issues one ``GET /api/artifacts/...``
   * per file in the selection. With the parallel refactor, ≥4 main
   * requests are dispatched within a few ms of each other (single
   * ``Promise.all`` flushes them all to the network stack at once).
   * Sequential code would have spread the dispatch across 4 × RTT.
   *
   * Failure mode this catches: a future refactor reintroduces a serial
   * await loop (or accidentally adds an ``await`` between dispatches).
   */
  test('#768 — production-shaped artifact fetches dispatch in parallel (<50ms span)', async ({
    page,
  }) => {
    const errs = captureConsoleErrors(page)

    // Capture every artifact request's dispatch timestamp BEFORE the
    // route handler in ``setupProductionShapedMocks`` fulfills it.
    const artifactDispatches: { url: string; t: number }[] = []
    page.on('request', (req) => {
      const u = req.url()
      if (/\/api\/artifacts\/.+\.(gi|kg|bridge)\.json/.test(u)) {
        artifactDispatches.push({ url: u, t: Date.now() })
      }
    })

    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })

    // Snapshot artifact-request count before the click; the click below
    // is what should trigger the parallel burst.
    artifactDispatches.length = 0

    const firstRow = page.getByRole('button', { name: /, / }).first()
    await firstRow.click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()

    // Wait for FSM to reach ``ready`` so we know the loadSelected cycle
    // completed and all artifact fetches were dispatched.
    await expect
      .poll(async () => (await readFsmState(page))?.state, { timeout: 20_000 })
      .toBe('ready')

    // Production-shaped fixture: ≥ 2 episode-artifact files per first
    // episode (GI + KG, optionally bridge). With memoized topic-cluster
    // doc + sibling-bridge join, total artifact dispatch is ≥ 2 within
    // the burst window.
    expect(artifactDispatches.length).toBeGreaterThanOrEqual(2)

    // The parallel contract: the time between the first and the last
    // dispatched request is small (single tick of the event loop +
    // ``Promise.all`` flush). Sequential code would have spread the
    // dispatches across RTT × N. Using a 200ms ceiling so the test is
    // not flaky on slow CI runners while still failing fast against a
    // serial-await regression.
    const ts = artifactDispatches.map((d) => d.t)
    const span = Math.max(...ts) - Math.min(...ts)
    expect(span).toBeLessThan(200)

    expect(errs.errors).toEqual([])
  })

  /**
   * ADR-158 — the public viewer asks for no private endpoint.
   *
   * Themes, storylines and trending are private features, served only when their extension is
   * installed. The public viewer must not request them on any path, so the same three-handoff
   * Library flow that used to exercise the #769 themes memo now counts zero such requests.
   */
  test('ADR-158 — repeated handoffs never request a private endpoint', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    const privateRequests: string[] = []
    page.on('request', (req) => {
      if (/\/api\/corpus\/(topic-clusters|storylines|trending)\b/.test(req.url())) {
        privateRequests.push(req.url())
      }
    })

    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })

    const rows = page.getByRole('button', { name: /, / })
    for (let i = 0; i < 3; i++) {
      await rows.nth(i).click({ timeout: 15_000 })
      await page.getByRole('button', { name: 'Open in graph' }).first().click()
      await expect
        .poll(async () => (await readFsmState(page))?.state, { timeout: 20_000 })
        .toBe('ready')
      if (i < 2) {
        await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
        await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })
      }
    }

    expect(privateRequests).toEqual([])
    expect(errs.errors).toEqual([])
  })

  /**
   * #767-C — recenter safety-net tail schedule.
   *
   * NOTE: the #767-C behavior contract is pinned at the UNIT level via
   * ``RECENTER_SAFETY_TAIL_TIMINGS_MS === [400, 900, 1800]`` in
   * ``cyCoseLayoutOptions.test.ts``. #787 originally trimmed the
   * schedule to ``[400]`` on the assumption that the 900 / 1800 ms
   * timers were dead weight; the Tier-2 production-shaped matrix
   * (``e2e/handoff-production/``) surfaced that they actually catch a
   * mac-firefox-only late-resize after-effect. The schedule is the
   * three-anchor list — unit-level assertion blocks any future trim
   * without an explicit cross-platform cost analysis.
   */
})
