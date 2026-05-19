/**
 * Tier-2 cold-start matrix against the production-shaped fixture.
 *
 * Mirrors the rows in ``e2e/handoff/cold-start.spec.ts`` but exercises
 * them against a real-corpus-extracted fixture (9 episodes, 5 feeds,
 * 150 topic clusters, GI+KG → ~270 cy nodes after merge). The same
 * 6-point ``assertHandoffApplied`` contract applies; bugs that depend
 * on graph scale or layout timing surface here.
 *
 * RFC-086 / ADR-095.
 */

import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'
import {
  assertFsmEventEnvelope,
  assertHandoffApplied,
  captureConsoleErrors,
} from '../handoff/_handoff-helpers'
import {
  fixtureEpisodes,
  setupProductionShapedMocks,
} from './_helpers'

test.describe('Handoff matrix § Tier 2 — Cold-start (production-shaped)', () => {
  test('P1.1 — Library row "Open in graph" (production-shaped)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })

    // First episode in the fixture.
    const first = fixtureEpisodes()[0]!
    const epTitle = (
      (await page.evaluate(async () => {
        try {
          const r = await fetch(
            '/api/corpus/episodes?path=/mock/production-shaped&limit=10',
          )
          const j = await r.json()
          return j.items?.[0]?.episode_title ?? null
        } catch {
          return null
        }
      })) as string | null
    ) ?? ''

    // Library renders rows with aria-label "{episode_title}, {feed_title}".
    // Click the first row by its title prefix.
    const firstRowBtn = page
      .getByRole('button', { name: new RegExp(epTitle.slice(0, 20)) })
      .first()
    await firstRowBtn.click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()

    // Same 6-point contract as Tier 1; resolves to the canonical
    // ``__unified_ep__:<UUID>`` cy id under the merged GI+KG graph.
    await assertHandoffApplied(page, `__unified_ep__:${first.episode_id}`, {
      errors: errs,
      episodePanelTitle: epTitle.length ? epTitle : undefined,
      // Real corpus has 270+ nodes; allow more breathing room for camera
      // pan than the tiny fast-matrix fixture.
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })
  })

  test('P1.6 — Episode panel "Open in graph" (production-shaped)', async ({ page }) => {
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible({ timeout: 15_000 })

    const first = fixtureEpisodes()[0]!
    const firstRow = page.getByRole('button', { name: /, / }).first()
    await firstRow.click({ timeout: 15_000 })
    await page.getByRole('button', { name: 'Open in graph' }).first().click()

    await assertHandoffApplied(page, `__unified_ep__:${first.episode_id}`, {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })
  })

  test('P1.5 — Search via dev-hook envelope (production-shaped)', async ({ page }) => {
    // S1 via dev hook: same FSM-event contract as Tier 1 H1.5. Search UI
    // requires a real vector index that the fixture doesn't include;
    // dev-hook drives the same envelope shape that SearchPanel produces
    // post-F1.6 wiring.
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page, { search: true })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await page.waitForTimeout(800)

    await page.evaluate(() => {
      const w = window as unknown as { __GIKG_FSM_EVENT_LOG__?: unknown[] }
      w.__GIKG_FSM_EVENT_LOG__ = []
    })
    await page.evaluate(() => {
      const store = (
        window as unknown as {
          __GIKG_HANDOFF_STORE__?: {
            handoffRequested: (env: Record<string, unknown>) => void
          }
        }
      ).__GIKG_HANDOFF_STORE__
      store?.handoffRequested({
        kind: 'topic',
        cyId: 'topic:science-research',
        source: 'search',
        loadSource: 'subject-external',
        camera: { kind: 'center-on-target' },
      })
    })
    await page.waitForTimeout(400)
    await assertFsmEventEnvelope(page, {
      type: 'handoffRequested',
      source: 'search',
      kind: 'topic',
      loadSource: 'subject-external',
      cameraKind: 'center-on-target',
      errors: errs,
    })
  })
})
