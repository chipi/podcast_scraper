/**
 * Topic Entity View (TEV) Playwright spec — coverage gap surfaced by
 * #678 PR-A audit.
 *
 * TEV is the SubjectRail panel that renders when
 * ``subject.kind === 'topic'``. Entry point (per E2E_SURFACE_MAP §224):
 *
 *   * Digest topic-title click → calls ``subject.focusTopic(graph_topic_id)``
 *     alongside the existing graph-open flow.
 *
 * Smoke contract (per E2E_SURFACE_MAP):
 *
 *   - ``topic-entity-view`` root visible
 *   - ``topic-entity-view-kind`` shows "Topic" or "Entity"
 *   - ``topic-entity-view-name`` shows the topic name
 *   - ``topic-entity-view-stats`` shows the dated-mentions sentence
 *   - When mentions exist: ``topic-entity-view-mentions`` list appears;
 *     when empty: ``topic-entity-view-empty`` placeholder
 *   - Action buttons ``topic-entity-view-go-graph`` /
 *     ``topic-entity-view-prefill-search`` present
 */

import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput } from './helpers'

const TOPIC_NAME = 'Mock Test Topic'

// Minimal corpus_digest mock that rolls up one mock topic so the
// digest topic-title click entry point is wired.
const DIGEST_RESPONSE = {
  kind: 'corpus_digest',
  recent_rows: [
    {
      title: 'Mock Episode for TEV',
      episode_id: 'ep-tev-mock-1',
      published_at: '2026-04-15T00:00:00Z',
      topic_clusters: [
        {
          graph_topic_id: 'topic-mock-1',
          name: TOPIC_NAME,
          weight: 0.9,
        },
      ],
    },
  ],
  topics: [
    {
      graph_topic_id: 'topic-mock-1',
      name: TOPIC_NAME,
      mention_count: 1,
      episode_count: 1,
    },
  ],
}

test.describe('Topic / Entity rail panel (TEV)', () => {
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

    await page.route('**/api/corpus/digest?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(DIGEST_RESPONSE),
      })
    })

    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
  })

  test('digest topic title → TEV renders the contract surface', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()

    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()

    // Wait for digest content to render then click the topic title.
    await page.getByText(TOPIC_NAME, { exact: false }).first().waitFor({ timeout: 10_000 })
    await page.getByText(TOPIC_NAME, { exact: false }).first().click()

    // TEV contract surface (E2E_SURFACE_MAP §224).
    const view = page.getByTestId('topic-entity-view')
    await expect(view).toBeVisible({ timeout: 10_000 })

    await expect(page.getByTestId('topic-entity-view-kind')).toContainText(/topic|entity/i)
    await expect(page.getByTestId('topic-entity-view-name')).toContainText(TOPIC_NAME)
    await expect(page.getByTestId('topic-entity-view-stats')).toBeVisible()

    // Either mentions list OR empty placeholder must render.
    const mentions = view.getByTestId('topic-entity-view-mentions')
    const empty = view.getByTestId('topic-entity-view-empty')
    const mentionsCount = await mentions.count()
    const emptyCount = await empty.count()
    expect(mentionsCount + emptyCount).toBeGreaterThan(0)

    // Action buttons present.
    await expect(page.getByTestId('topic-entity-view-go-graph')).toBeVisible()
    await expect(page.getByTestId('topic-entity-view-prefill-search')).toBeVisible()
  })
})
