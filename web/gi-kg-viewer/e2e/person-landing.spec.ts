/**
 * Person Landing Playwright spec — coverage gap surfaced by #678 PR-A audit;
 * extended for #1048 shared-component shell.
 *
 * Person Landing is the SubjectRail panel that renders when
 * ``subject.kind === 'person'``. Entry points (per E2E_SURFACE_MAP §223):
 *
 *   * Explore "Top speakers" rollup → ``explore-top-speaker-link``
 *   * Search result supporting-quote speaker → ``search-result-speaker-link``
 *
 * Both call ``subject.focusPerson(speaker_id)`` which sets the rail's
 * subject store. This spec uses the explore-rollup entry path because it's
 * the more deterministic of the two — it doesn't depend on a search index.
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

import { readFileSync } from 'node:fs'
import { expect, test } from '@playwright/test'
import { GI_SAMPLE_FIXTURE } from './fixtures'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

const artifactJson = readFileSync(GI_SAMPLE_FIXTURE, 'utf-8')

// Realistic corpus id: GI speaker_ids are ``person:``-prefixed (e.g.
// ``person:alice-hayes``), which is what NodeDetail's off-slice
// ``inferredKindFromId`` keys on to render the Person node view for a
// focused speaker that isn't in the current graph slice. A bare id never
// matches the convention, so the rail would stay an empty "Node" shell.
const SPEAKER_ID = 'person:speaker-mock-1'
const SPEAKER_NAME = 'Marko Mock-Guest'

// Minimal Explore mock that rolls up one mock speaker so the
// ``explore-top-speaker-link`` entry point is rendered.
const EXPLORE_RESPONSE = {
  kind: 'explore',
  data: {
    episodes_searched: 1,
    summary: {
      insight_count: 1,
      grounded_insight_count: 1,
      quote_count: 1,
      episode_count: 1,
      speaker_count: 1,
      topic_count: 0,
    },
    insights: [
      {
        insight_id: 'ins:person-landing-mock',
        text: `${SPEAKER_NAME} discussed the rolling shutter problem.`,
        grounded: true,
        confidence: 0.91,
        episode: { episode_id: 'ep-mock-1', title: 'Mock Episode' },
        supporting_quotes: [
          {
            text: 'Rolling shutter is the silent killer of low-light video.',
            speaker: { speaker_id: SPEAKER_ID, name: SPEAKER_NAME },
          },
        ],
      },
    ],
    top_speakers: [
      {
        speaker_id: SPEAKER_ID,
        name: SPEAKER_NAME,
        quote_count: 1,
        episode_count: 1,
      },
    ],
  },
}

test.describe('Person Landing rail panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          explore_api: true,
        }),
      })
    })

    await page.route('**/api/explore?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(EXPLORE_RESPONSE),
      })
    })

    // Mock one GI artifact so the Graph workspace auto-loads (Fit
    // toolbar button is the gate the spec then waits on). The Top
    // speakers rollup itself comes from the explore mock, but the
    // ``left-panel-enter-explore`` button only renders inside the
    // Graph workspace, so a graph must be loaded first.
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          artifacts: [
            {
              name: 'ci_sample.gi.json',
              relative_path: 'metadata/ci_sample.gi.json',
              kind: 'gi',
              size_bytes: artifactJson.length,
              mtime_utc: '2026-04-18T12:00:00Z',
              publish_date: '2026-04-18',
            },
          ],
        }),
      })
    })

    await page.route('**/api/artifacts/metadata/ci_sample.gi.json?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: artifactJson,
      })
    })
  })

  test('explore Top speakers → Person Landing renders the contract surface', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })

    // Open Explore + submit so the Top speakers rollup populates.
    await page.getByTestId('left-panel-enter-explore').click()
    await page.getByRole('heading', { name: /Explore & query/i }).waitFor({ state: 'visible' })
    await page.getByTestId('explore-filtered-submit').click()

    // Wait for the explore result list to render then click the speaker
    // link in the Top speakers rollup.
    await page.getByText(SPEAKER_NAME, { exact: false }).first().waitFor({ timeout: 10_000 })
    await page.getByTestId('explore-top-speaker-link').first().click()

    // Person node view, folded embedded into the NodeDetail rail (Details tab).
    const rail = page.getByTestId('graph-node-detail-rail')
    await expect(page.getByTestId('person-landing-view')).toBeVisible({ timeout: 10_000 })

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
            { id: 'insight:1', type: 'insight', text: 'A stated position on policy.', show_id: '', episode_id: 'e1' },
            { id: 'insight:2', type: 'insight', text: 'Another stated position.', show_id: '', episode_id: 'e1' },
          ],
          error: null,
        }),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await page.getByTestId('left-panel-enter-explore').click()
    await page.getByRole('heading', { name: /Explore & query/i }).waitFor({ state: 'visible' })
    await page.getByTestId('explore-filtered-submit').click()
    await page.getByText(SPEAKER_NAME, { exact: false }).first().waitFor({ timeout: 10_000 })
    await page.getByTestId('explore-top-speaker-link').first().click()
    await expect(page.getByTestId('person-landing-view')).toBeVisible({ timeout: 10_000 })

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
  async function gotoPersonLanding(page: import('@playwright/test').Page): Promise<void> {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await page.getByTestId('left-panel-enter-explore').click()
    await page.getByRole('heading', { name: /Explore & query/i }).waitFor({ state: 'visible' })
    await page.getByTestId('explore-filtered-submit').click()
    await page.getByText(SPEAKER_NAME, { exact: false }).first().waitFor({ timeout: 10_000 })
    await page.getByTestId('explore-top-speaker-link').first().click()
    await expect(page.getByTestId('person-landing-view')).toBeVisible({ timeout: 10_000 })
  }

  test('#1055: Connections section renders topics + co-speakers chips', async ({ page }) => {
    await page.route('**/api/relational/topics**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          subject: SPEAKER_ID,
          results: [
            { id: 'topic:markets', type: 'topic', text: 'Markets', show_id: '', episode_id: '' },
            { id: 'topic:rates', type: 'topic', text: 'Interest rates', show_id: '', episode_id: '' },
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
            { id: 'person:rob', type: 'person', text: 'Rob Armstrong', show_id: '', episode_id: '' },
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
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(empty) })
    })
    await page.route('**/api/relational/co-speakers**', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(empty) })
    })

    await gotoPersonLanding(page)

    await expect(page.getByTestId('person-landing-topics-empty')).toBeVisible()
    await expect(page.getByTestId('person-landing-co-speakers-empty')).toBeVisible()
  })
})
