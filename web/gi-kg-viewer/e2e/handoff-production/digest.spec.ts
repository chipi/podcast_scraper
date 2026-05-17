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

  // P1.2-bucket — Topic-band headline pill targeting an aggregation bucket
  //
  // **Architectural change during real-corpus validation:** the headline
  // topic-band pill (`Science & research`, `Technology`, `Business & markets`)
  // is no longer clickable. These are editorial defaults from
  // ``DEFAULT_DIGEST_TOPICS`` in the server, not corpus-derived KG topics —
  // clicking them built envelopes targeting ``topic:<editorial-slug>`` that
  // wouldn't resolve in arbitrary corpora, producing the V2-class "silent
  // applied with bucket id" lie that this test originally caught.
  //
  // Rather than test the failure path, we now structurally prevent the
  // bug: ``DigestView.vue`` renders the headline as ``<span>`` (not
  // ``<button>``) so there's no click affordance. The "Search topic"
  // button next to each headline remains as the actionable
  // affordance (runs the FAISS query to surface real focusable hits).
  //
  // The test that previously validated the V2 fix-class is now N/A —
  // the affordance doesn't exist. Per-row CIL pills (covered by P1.2
  // above and Tier-3 P2.2 / P3.2) are the user-facing topic-pill paths.
})
