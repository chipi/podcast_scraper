/**
 * Tier-2 Digest pill matrix (production-shaped fixture). Targets the
 * V2 reproducer: in real corpus, the Digest topic pill set
 * ``subject.kind=topic`` but no cy node was selected. This row catches
 * subject↔cy resolution drift at scale.
 *
 * RFC-086.
 */

import { test, expect } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from '../helpers'
import {
  assertHandoffApplied,
  captureConsoleErrors,
} from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from './_helpers'

test.describe('Handoff matrix § Tier 2 — Digest pill (production-shaped)', () => {
  test('P1.2 — Digest recent-row CIL pill resolves to a real cy topic node', async ({ page }) => {
    // D1 in the catalog. Recent-row pills target ``cil_digest_topics[]``
    // entries which ARE present in the episode's GI artifact. Targets
    // the first recent row's first CIL topic (``topic:public-investment``
    // for the production-shaped fixture's first row).
    //
    // This is distinct from the **topic-band** pills at the top of the
    // Digest (``topic:science-research``, etc.) which are categorization
    // buckets, not real cy nodes — those correctly surface as
    // ``handoffFailed`` per the V2 fix.
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 15_000 })

    // Recent-row pill — match by the specific label rather than
    // ``first()`` so topic-band pills don't shadow.
    const pill = page
      .getByRole('button', { name: /Open graph for topic: public investment/i })
      .first()
    await pill.waitFor({ state: 'visible', timeout: 15_000 })
    await pill.click()

    await assertHandoffApplied(page, 'g:topic:public-investment', {
      errors: errs,
      cameraCenterTolerance: 0.4,
      waitForReadyMs: 30_000,
    })
  })

  test('P1.2-bucket — Topic-band pill targeting bucket fails cleanly (V2 fix)', async ({ page }) => {
    // Pre-V2-fix: this would silently say "applied" with the bucket
    // cyId. Post-fix: FSM.handoffFailed surfaces the error strip so
    // the user knows the click went nowhere. Pinning the failure path
    // here means a future regression that re-introduces the silent
    // accept breaks this row.
    const errs = captureConsoleErrors(page)
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 15_000 })

    // Click the first topic-band pill (aggregation bucket, no cy node).
    const bandPill = page
      .getByRole('button', { name: /Open graph for topic Science/ })
      .first()
    await bandPill.waitFor({ state: 'visible', timeout: 15_000 })
    await bandPill.click()

    // Poll until lastResult lands (success or failure). The digest band
    // path appends artifacts for all 8 hits before firing the handoff,
    // so redraw can take 3-5 s on the production-shaped fixture.
    const fsm = await page.evaluate(
      async ({ maxMs }: { maxMs: number }) => {
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
          __GIKG_FSM__?: { state: string; lastResult: unknown }
        }
        return {
          state: w2.__GIKG_FSM__?.state ?? 'unknown',
          lastResult: w2.__GIKG_FSM__?.lastResult ?? null,
        }
      },
      { maxMs: 30_000 },
    )
    expect(fsm).not.toBeNull()
    expect(fsm!.lastResult?.status).toBe('failed')
    // Console errors from the API layer would be a real issue;
    // structured handoffFailed is logged at warn level, not error.
    expect(errs.errors).toEqual([])
  })
})
