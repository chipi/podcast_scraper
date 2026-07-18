/**
 * Tier-2 Digest pill matrix (production-shaped fixture). Targets the
 * V2 reproducer: in real corpus, the Digest topic pill set
 * ``subject.kind=topic`` but no cy node was selected. This row catches
 * subject↔cy resolution drift at scale.
 *
 * RFC-086.
 */

import { test, expect } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from '../helpers'
import {
  assertHandoffApplied,
  captureConsoleErrors,
} from '../handoff/_handoff-helpers'
import { setupProductionShapedMocks } from './_helpers'

test.describe('Handoff matrix § Tier 2 — Digest pill (production-shaped)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

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

  test('P1.2-bucket — headline topic-band pill renders as non-clickable (V2 regression guard)', async ({
    page,
  }) => {
    // **Architectural change during real-corpus validation:** the headline
    // topic-band pill (`Science & research`, `Technology`, `Business & markets`)
    // is no longer clickable. These are editorial defaults from
    // ``DEFAULT_DIGEST_TOPICS`` in the server, not corpus-derived KG topics —
    // clicking them built envelopes targeting ``topic:<editorial-slug>`` that
    // wouldn't resolve in arbitrary corpora, producing the V2-class "silent
    // applied with bucket id" lie that the original P1.2-bucket test caught.
    //
    // Rather than test the failure path, we structurally prevent the bug:
    // ``DigestView.vue`` renders the headline as ``<span>`` (not ``<button>``)
    // so there's no click affordance. This regression test ASSERTS that
    // structural state — if a future refactor silently re-introduces the
    // click target, this test goes red and the V2 fix doesn't decay.
    //
    // The "Search topic" button next to each headline remains as the
    // actionable affordance (runs the search query to surface real focusable
    // hits). Per-row CIL pills (covered by P1.2 above and Tier-3 P2.2 / P3.2)
    // are the user-facing topic-pill graph-handoff paths.
    await setupProductionShapedMocks(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/production-shaped')
    await expect(page.getByTestId('digest-root')).toBeVisible({ timeout: 15_000 })

    // Topic-bands region must render with the headline labels visible.
    const topicBands = page.getByRole('region', { name: 'Topic bands' })
    await expect(topicBands).toBeVisible({ timeout: 15_000 })

    // None of the headline bucket labels should be rendered as a clickable
    // graph-handoff button. The old affordance had aria-label
    // ``Open graph for topic <label> (top hit with GI or KG)`` — assert
    // there are zero such buttons.
    await expect(
      page.getByRole('button', { name: /^Open graph for topic .* \(top hit with GI or KG\)$/ }),
    ).toHaveCount(0)

    // The "Search topic" button must remain as the actionable surface
    // next to each headline (regression guard for the surviving path).
    await expect(
      topicBands.getByRole('button', { name: 'Search topic' }).first(),
    ).toBeVisible()
  })
})
