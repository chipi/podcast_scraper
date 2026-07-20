/**
 * Person Landing Playwright spec — coverage gap surfaced by #678 PR-A audit;
 * extended for #1048 shared-component shell; entry path re-targeted in Search
 * v3 §S1 stabilization pass (2026-07-20) from the retired Explore Top-speakers
 * rollup to a Search-hit → lifted-speaker-link click, which continues to work
 * against the merged Search launcher (compact-launcher shape after S1).
 *
 * Person Landing is the SubjectRail panel that renders when
 * ``subject.kind === 'person'``. Entry points:
 *
 *   * Search result with a ``lifted.speaker`` → ``search-result-lifted-speaker-link``
 *     (this spec; the shipped path today).
 *   * Rail launcher ``rail-search-in-person`` (Search v3 §S6, #1236; not yet
 *     landed — no test uses this yet).
 *
 * Both call ``subject.focusPerson(speaker_id)`` which sets the rail's subject
 * store. This spec uses the lifted-speaker-link entry because it's the surface
 * that ships today after S1's Explore merge; the tests exercise the
 * Person Landing view + its position/connections coverage the same way the
 * previous Explore-entry version did.
 *
 * Smoke contract (post node-view unification — the standalone Person rail is
 * retired; PersonLandingView is now folded ``embedded`` into NodeDetail's
 * rail, which owns the header + tabs, so PLV's own name header + internal
 * tablist + action buttons are hidden via ``v-if="!embedded"``):
 *
 *   - ``person-landing-view`` root visible (embedded in the Details rail tab)
 *   - the rail header (``graph-node-detail-rail``) titles it "Person"
 *   - NodeDetail rail tabs ``node-detail-rail-tab-details`` ("Details") and
 *     ``node-detail-rail-tab-position-tracker`` ("Positions") drive the view
 *   - Details is the default tab → ``person-landing-panel-profile`` visible
 *   - Switching to Positions reveals ``person-landing-positions-view``
 *   - Person node views only render for a ``person:``-prefixed id (the GI
 *     speaker_id convention) — NodeDetail's off-slice ``inferredKindFromId``
 *     keys on it so a focused speaker renders even outside the graph slice.
 */

import { expect, test } from '@playwright/test'
import { SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

// Realistic corpus id: GI speaker_ids are ``person:``-prefixed (e.g.
// ``person:alice-hayes``), which is what NodeDetail's off-slice
// ``inferredKindFromId`` keys on to render the Person node view for a
// focused speaker that isn't in the current graph slice. A bare id never
// matches the convention, so the rail would stay an empty "Node" shell.
const SPEAKER_ID = 'person:speaker-mock-1'
const SPEAKER_NAME = 'Marko Mock-Guest'
const SEARCH_QUERY = 'rolling shutter low-light video'

// Minimal /api/search response: one insight-tier hit with a ``lifted`` block
// carrying the mock speaker. That hit renders the ``search-result-lifted-speaker-link``
// which routes to ``subject.focusPerson(SPEAKER_ID)`` — the entry point this
// spec exercises.
const SEARCH_RESPONSE = {
  query: SEARCH_QUERY,
  query_type: 'raw_evidence',
  results: [
    {
      doc_id: 'insight:person-landing-mock',
      score: 0.91,
      text: `${SPEAKER_NAME} on rolling shutter as the silent killer of low-light video.`,
      source_tier: 'insight',
      metadata: {
        doc_type: 'insight',
        source_id: 'insight:person-landing-mock',
        episode_id: 'ep-mock-1',
        episode_title: 'Mock Episode',
        feed_id: 'sha256:mock',
        feed_title: 'Mock Feed',
        publish_date: '2026-04-18',
      },
      supporting_quotes: null,
      lifted: {
        insight: {
          id: 'insight:person-landing-mock',
          text: `${SPEAKER_NAME} on rolling shutter as the silent killer of low-light video.`,
          insight_type: 'observation',
          grounded: true,
        },
        speaker: {
          id: SPEAKER_ID,
          display_name: SPEAKER_NAME,
        },
        topic: null,
        quote: {
          timestamp_start_ms: 12000,
          timestamp_end_ms: 24500,
        },
      },
    },
  ],
  lift_stats: null,
  error: null,
  detail: null,
}

test.describe('Person Landing rail panel', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          search_api: true,
        }),
      })
    })

    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(SEARCH_RESPONSE),
      })
    })
  })

  /**
   * Shared entry: fills the search query, submits, waits for the lifted speaker
   * link, clicks it, waits for the Person Landing view to appear.
   *
   * The compact launcher lives in the LeftPanel on every main tab post-S1, so
   * no tab switch is needed — unlike the retired Explore path which only rendered
   * inside the Graph workspace.
   */
  async function gotoPersonLanding(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.locator('#search-q').waitFor({ state: 'visible', timeout: 30_000 })
    await expect(page.locator('#search-q')).toBeEnabled({ timeout: 10_000 })
    await page.locator('#search-q').fill(SEARCH_QUERY)
    await page
      .locator('form#semantic-search-form')
      .getByRole('button', { name: /^Search$/ })
      .click()
    await page.getByTestId('search-result-lifted-speaker-link').first().waitFor({
      state: 'visible',
      timeout: 10_000,
    })
    await page.getByTestId('search-result-lifted-speaker-link').first().click()
    await expect(page.getByTestId('person-landing-view')).toBeVisible({ timeout: 10_000 })
  }

  test('Search hit → lifted speaker link → Person Landing renders the contract surface', async ({
    page,
  }) => {
    await gotoPersonLanding(page)

    // Person node view, folded embedded into the NodeDetail rail (Details tab).
    const rail = page.getByTestId('graph-node-detail-rail')

    // The rail (not PLV) owns the header; an off-slice ``person:`` id titles
    // it "Person" via NodeDetail's inferredKindFromId.
    await expect(rail.getByRole('heading').first()).toContainText('Person')

    // NodeDetail rail tabs drive the person view (PLV's own internal tablist is
    // hidden in embedded mode). Details is default → the profile panel renders.
    await expect(page.getByTestId('node-detail-rail-tab-details')).toHaveText(/Details/)
    await expect(page.getByTestId('person-landing-panel-profile')).toBeVisible()

    // Positions rail tab reveals the positions lens (stated positions +
    // Position Tracker), carried by the dedicated positions PLV instance.
    await page.getByTestId('node-detail-rail-tab-position-tracker').click()
    await expect(page.getByTestId('person-landing-positions-view')).toBeVisible()

    // Back to Details re-shows the profile panel.
    await page.getByTestId('node-detail-rail-tab-details').click()
    await expect(page.getByTestId('person-landing-panel-profile')).toBeVisible()
  })

  test('FR4.1: stated positions from the relational layer render in the Positions rail tab', async ({
    page,
  }) => {
    await page.route('**/api/relational/positions**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: SPEAKER_ID,
          results: [
            {
              id: 'insight:1',
              type: 'insight',
              text: 'A stated position on policy.',
              show_id: '',
              episode_id: 'e1',
            },
            {
              id: 'insight:2',
              type: 'insight',
              text: 'Another stated position.',
              show_id: '',
              episode_id: 'e1',
            },
          ],
          error: null,
        }),
      })
    })

    await gotoPersonLanding(page)

    // Post node-view unification: stated positions live in the Positions rail
    // tab → "All positions" lens (the default lens is "By topic"), not the
    // Details/profile panel.
    await page.getByTestId('node-detail-rail-tab-position-tracker').click()
    await page.getByTestId('person-landing-positions-lens-all').click()
    const stated = page.getByTestId('person-landing-stated')
    await expect(stated).toBeVisible()
    await expect(stated.getByTestId('person-landing-stated-row')).toHaveCount(2)
    await expect(stated).toContainText('A stated position on policy.')
  })

  // #1055: the Profile tab's "Connections" section consumes the new relational
  // routes (/relational/topics + /relational/co-speakers). These assert the viewer
  // wiring end-to-end (the real graph traversal is covered by the Python tests).

  test('#1055: Connections section renders topics + co-speakers chips', async ({ page }) => {
    await page.route('**/api/relational/topics**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: SPEAKER_ID,
          results: [
            { id: 'topic:markets', type: 'topic', text: 'Markets', show_id: '', episode_id: '' },
            {
              id: 'topic:rates',
              type: 'topic',
              text: 'Interest rates',
              show_id: '',
              episode_id: '',
            },
          ],
          error: null,
        }),
      })
    })
    await page.route('**/api/relational/co-speakers**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: SPEAKER_ID,
          results: [
            {
              id: 'person:rob',
              type: 'person',
              text: 'Rob Armstrong',
              show_id: '',
              episode_id: '',
            },
          ],
          error: null,
        }),
      })
    })

    await gotoPersonLanding(page)

    const topics = page.getByTestId('person-landing-topics')
    await expect(topics).toBeVisible()
    await expect(topics.getByTestId('person-landing-topic-chip')).toHaveCount(2)
    await expect(topics).toContainText('Interest rates')

    const coSpeakers = page.getByTestId('person-landing-co-speakers')
    await expect(coSpeakers).toBeVisible()
    await expect(coSpeakers.getByTestId('person-landing-co-speaker-chip')).toHaveCount(1)
    await expect(coSpeakers).toContainText('Rob Armstrong')
  })

  test('#1055: Connections section shows honest empty states when no connectivity', async ({
    page,
  }) => {
    const empty = { subject: SPEAKER_ID, results: [], error: null }
    await page.route('**/api/relational/topics**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(empty),
      })
    })
    await page.route('**/api/relational/co-speakers**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(empty),
      })
    })

    await gotoPersonLanding(page)

    await expect(page.getByTestId('person-landing-topics-empty')).toBeVisible()
    await expect(page.getByTestId('person-landing-co-speakers-empty')).toBeVisible()
  })
})
