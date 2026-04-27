/**
 * Person Landing Playwright spec — coverage gap surfaced by #678 PR-A audit.
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
 * Smoke contract (per E2E_SURFACE_MAP):
 *
 *   - ``person-landing-view`` root visible
 *   - ``person-landing-view-name`` shows the speaker's name
 *   - ``role="tablist"`` with two tabs:
 *     - ``person-landing-tab-profile`` (default selected)
 *     - ``person-landing-tab-positions``
 *   - Profile panel ``person-landing-panel-profile`` is visible
 *   - Switching to Positions panel reveals
 *     ``person-landing-panel-positions``
 *   - Action buttons ``person-landing-go-graph`` /
 *     ``person-landing-prefill-search`` are present
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

const SPEAKER_ID = 'speaker-mock-1'
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

    // Empty artifacts list — Person Landing only needs the explore data
    // for the rollup link entry, and reads person-mention details on
    // demand from the subject store. Detailed mention chart data is
    // out of scope for this smoke spec.
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
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

    // Person Landing contract surface (E2E_SURFACE_MAP §223).
    const view = page.getByTestId('person-landing-view')
    await expect(view).toBeVisible({ timeout: 10_000 })

    await expect(page.getByTestId('person-landing-view-name')).toContainText(SPEAKER_NAME)

    const tablist = view.getByRole('tablist')
    await expect(tablist).toBeVisible()
    await expect(page.getByTestId('person-landing-tab-profile')).toBeVisible()
    await expect(page.getByTestId('person-landing-tab-positions')).toBeVisible()

    // Profile is default; switching to Positions reveals the positions panel.
    await expect(page.getByTestId('person-landing-panel-profile')).toBeVisible()
    await page.getByTestId('person-landing-tab-positions').click()
    await expect(page.getByTestId('person-landing-panel-positions')).toBeVisible()

    // Action buttons present (data-driven assertions on edge counts /
    // mentions chart deliberately out of scope for this smoke spec —
    // those need a full bridge fixture).
    await expect(page.getByTestId('person-landing-go-graph')).toBeVisible()
    await expect(page.getByTestId('person-landing-prefill-search')).toBeVisible()
  })
})
