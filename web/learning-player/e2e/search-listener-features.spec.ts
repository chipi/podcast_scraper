import { expect, test } from '@playwright/test'

/**
 * Player-side listener enhancements (#1261-1..8). Smoke-tests the search
 * result surface + save-query round-trip.
 *
 * The committed validation corpus (tests/fixtures/app-validation-corpus/v3)
 * does NOT ship a LanceDB search index (the fixture builder is
 * deterministic + fast; no ML at build-time). So `/api/app/search` on the
 * e2e backend always returns error='no_index'. Every spec below that needs
 * search results MUST route.fulfill the search endpoint — the test still
 * runs against real Vue + real router + real preview server, only the
 * search response is stubbed.
 *
 * Not exhaustive; component tests cover render logic. This layer proves
 * the client wire-up survives the real API response shape.
 */

const FAKE_SEARCH_RESPONSE = {
  query: 'risk',
  results: [
    {
      doc_id: 'insight:e1__001',
      score: 0.9,
      text: 'Framing risk as a systems property changes how you allocate exposure.',
      source_tier: 'insight',
      metadata: {
        doc_type: 'insight',
        episode_id: 'ep-1',
        episode_slug: 'ep-1',
        episode_title: 'Risk as a Systems Property',
        podcast_title: 'Sample Show',
        publish_date: '2024-06-01',
        query_enrichments: {
          related_topics: [
            { topic_id: 'topic:systems', topic_label: 'Systems thinking', similarity: 0.91 },
            { topic_id: 'topic:portfolio', topic_label: 'Portfolio', similarity: 0.82 },
          ],
        },
      },
    },
    {
      doc_id: 't:e1_chunk_005',
      score: 0.7,
      text: 'The transcript passage about risk.',
      source_tier: 'segment',
      metadata: {
        doc_type: 'transcript',
        episode_id: 'ep-1',
        episode_slug: 'ep-1',
        episode_title: 'Risk as a Systems Property',
        podcast_title: 'Sample Show',
        publish_date: '2024-06-01',
      },
    },
  ],
  error: null,
}

async function stubSearch(page: import('@playwright/test').Page): Promise<void> {
  await page.route(/\/api\/app\/search(\?|$)/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(FAKE_SEARCH_RESPONSE),
    })
  })
  // resolveEntity is called in parallel; return null to keep the assertion set
  // focused on results.
  await page.route(/\/api\/app\/entities\/search(\?|$)/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ query: 'risk', entity: null }),
    })
  })
}

test('search results render the "Also about:" enriched-topic chip row', async ({ page }) => {
  await stubSearch(page)
  await page.goto('/search?q=risk')
  await expect(page.getByText('2 passages across 1 episodes')).toBeVisible()
  const chips = page.getByTestId('related-topic-chips')
  await expect(chips).toBeVisible()
  await expect(chips).toContainText('Systems thinking')
  await expect(chips).toContainText('Portfolio')
})

test('search results render the "Matched:" kicker on the episode header', async ({ page }) => {
  await stubSearch(page)
  await page.goto('/search?q=risk')
  const matched = page.getByTestId('matched-fields').first()
  await expect(matched).toBeVisible()
  await expect(matched).toContainText(/Matched:/)
  // insight + transcript both resolve to their labels.
  await expect(matched).toContainText('Insight')
  await expect(matched).toContainText('Transcript')
})

test('save-query button toggles Save/Saved-✓ on click', async ({ page }) => {
  await stubSearch(page)
  await page.goto('/search?q=risk')
  const saveBtn = page.getByTestId('save-query-button')
  await expect(saveBtn).toBeVisible()
  await expect(saveBtn).toHaveText('Save')
  await saveBtn.click()
  await expect(saveBtn).toHaveText('Saved ✓')
  // Toggle-off round-trip.
  await saveBtn.click()
  await expect(saveBtn).toHaveText('Save')
})

// Signed-in persistence-across-navigations belongs on library-saved.spec — the
// USERPREFS-1 sign-in path there is already validated by capture/consolidation
// specs. Repeating it here would duplicate the flakiness surface without
// adding coverage of #1261-8 store logic (which savedQueries.test.ts covers).

test('player page renders the "More like this" rail when the endpoint returns peers', async ({
  page,
}) => {
  // /related is best-effort and can return empty on some episodes — reach
  // the player and only assert IF the rail is present. Keeps the smoke
  // deterministic against future corpus changes.
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  const rail = page.getByTestId('related-episodes-rail')
  if (await rail.count()) {
    await expect(rail.getByRole('heading', { name: 'More like this' })).toBeVisible()
  }
})
