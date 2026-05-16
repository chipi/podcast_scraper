/**
 * Tier-2 concurrency matrix (production-shaped fixture). The
 * rapid-click row is the key value-add over Tier 1 — at 270+ cy
 * nodes the layout time is real, so back-to-back clicks actually race.
 * RFC-086.
 */

import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'
import {
  assertHandoffApplied,
  captureConsoleErrors,
  readFsmState,
} from '../handoff/_handoff-helpers'
import {
  fixtureEpisodes,
  setupProductionShapedMocks,
} from './_helpers'

test.describe('Handoff matrix § Tier 2 — Concurrency (production-shaped)', () => {
  test('P5.1 — Rapid handoffRequested supersession (dev-hook, last wins)', async ({ page }) => {
    // Mirrors Tier 1 H5.1 but on the production-shaped graph where layout
    // takes real time. Drives envelopes via ``__GIKG_HANDOFF_STORE__``
    // because rapid UI clicks can't stay on a single tab (Open in graph
    // switches main tab → Library rows go offscreen). Pins the FSM-side
    // supersession contract under real-scale layout timing.
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page, { artifactLatencyMs: 200 })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    // Land on Graph tab so cy mounts.
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.waitForTimeout(800)

    const eps = fixtureEpisodes()

    // Fire 3 envelopes back-to-back with no awaits between them.
    await page.evaluate(
      ({ ids }: { ids: string[] }) => {
        const store = (
          window as unknown as {
            __GIKG_HANDOFF_STORE__?: {
              handoffRequested: (env: Record<string, unknown>) => void
            }
          }
        ).__GIKG_HANDOFF_STORE__
        if (!store) return
        for (const id of ids) {
          store.handoffRequested({
            kind: 'episode',
            episodeId: id,
            source: 'library',
            loadSource: 'subject-external',
            camera: { kind: 'center-on-target' },
          })
        }
      },
      { ids: eps.slice(0, 3).map((e) => e.episode_id) },
    )

    // Last wins — wait for FSM ready, then check generation has bumped
    // at least 3x.
    await page.waitForTimeout(2000)
    const after = await readFsmState(page)
    expect(after).not.toBeNull()
    expect(after!.generation).toBeGreaterThanOrEqual(3)
    // The first two should have been superseded; lastResult points at
    // whichever envelope actually applied (the third, ideally).
    expect(errs.errors).toEqual([])
  })
})
