import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE } from './helpers'

test.describe('Corpus Digest tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
        }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          artifacts: [],
        }),
      })
    })
    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feeds: [{ feed_id: 'f1', display_title: 'Mock Feed Show', episode_count: 1 }],
        }),
      })
    })
    await page.route('**/api/corpus/episodes/detail**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          metadata_relative_path: 'metadata/ep1.metadata.json',
          feed_id: 'f1',
          episode_id: 'e1',
          episode_title: 'Digest Episode Alpha',
          publish_date: '2024-06-05',
          summary_title: 'Digest summary',
          summary_bullets: ['First bullet'],
          summary_text: null,
          gi_relative_path: 'metadata/ep1.gi.json',
          kg_relative_path: 'metadata/ep1.kg.json',
          has_gi: true,
          has_kg: false,
        }),
      })
    })
    await page.route(
      (url) => {
        const p = new URL(url).pathname.replace(/\/$/, '')
        return p === '/api/corpus/episodes'
      },
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            path: '/mock/corpus',
            feed_id: null,
            items: [
              {
                metadata_relative_path: 'metadata/ep1.metadata.json',
                feed_id: 'f1',
                feed_display_title: 'Mock Feed Show',
                topics: ['First bullet'],
                summary_title: 'Digest summary',
                summary_bullets_preview: ['First bullet'],
                summary_preview: 'Digest summary — First bullet',
                episode_id: 'e1',
                episode_title: 'Digest Episode Alpha',
                publish_date: '2024-06-05',
              },
            ],
            next_cursor: null,
          }),
        })
      },
    )
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: true,
          reason: null,
          stats: {
            total_vectors: 1,
            doc_type_counts: {},
            feeds_indexed: ['f1'],
            embedding_model: 'mock',
            embedding_dim: 8,
            last_updated: '2024-01-01T00:00:00Z',
            index_size_bytes: 0,
          },
          reindex_recommended: false,
        }),
      })
    })
    await page.route('**/api/corpus/episodes/similar**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          source_metadata_relative_path: 'metadata/ep1.metadata.json',
          query_used: 'Digest summary First bullet',
          items: [],
          error: null,
          detail: null,
        }),
      })
    })
    await page.route('**/api/corpus/digest**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          window: '7d',
          window_start_utc: '2024-06-01T00:00:00Z',
          window_end_utc: '2024-06-08T00:00:00Z',
          compact: false,
          rows: [
            {
              metadata_relative_path: 'metadata/ep1.metadata.json',
              feed_id: 'f1',
              feed_display_title: 'Mock Feed Show',
              episode_id: 'e1',
              episode_title: 'Digest Episode Alpha',
              publish_date: '2024-06-05',
              summary_title: 'Digest summary',
              summary_bullets_preview: ['First bullet'],
              summary_preview: 'Digest summary — First bullet',
              gi_relative_path: 'metadata/ep1.gi.json',
              kg_relative_path: 'metadata/ep1.kg.json',
              has_gi: true,
              has_kg: false,
            },
          ],
          topics: [
            {
              topic_id: 't1',
              label: 'Mock Topic Band',
              query: 'climate science',
              hits: [
                {
                  metadata_relative_path: 'metadata/ep1.metadata.json',
                  episode_title: 'Digest Episode Alpha',
                  feed_id: 'f1',
                  feed_display_title: 'Mock Feed Show',
                  publish_date: '2024-06-05',
                  score: 0.91,
                  summary_preview: 'Digest summary — First bullet',
                  episode_id: 'e1',
                  gi_relative_path: 'metadata/ep1.gi.json',
                  kg_relative_path: 'metadata/ep1.kg.json',
                  has_gi: true,
                  has_kg: false,
                },
              ],
            },
          ],
          topics_unavailable_reason: null,
        }),
      })
    })
  })

  test('shows mocked digest rows and topic band; Search topic prefills query', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    const digestRecentCard = page.getByRole('button', {
      name: 'Digest Episode Alpha, Mock Feed Show',
      exact: true,
    })
    await expect(digestRecentCard).toBeVisible()
    await expect(digestRecentCard.getByText('Digest summary — First bullet')).toBeVisible()
    await expect(
      page.getByTestId('digest-root').getByRole('button', { name: 'First bullet' }),
    ).toHaveCount(0)
    await expect(
      page.getByRole('button', { name: 'Open top hit for topic Mock Topic Band in graph' }),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Search topic' }).first().click()
    await expect(page.locator('#search-q')).toHaveValue('climate science')
    await expect(
      page.getByRole('textbox', { name: 'Since (date)' }),
    ).toHaveValue('2024-06-01')
  })

  test('digest episode cards omit graph/search actions (Episode rail has them)', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    const digestRoot = page.getByTestId('digest-root')
    await expect(digestRoot.getByRole('button', { name: 'Open in graph' })).toHaveCount(0)
    await expect(
      digestRoot.getByRole('button', { name: 'Prefill semantic search' }),
    ).toHaveCount(0)
  })

  test('Open Library switches to Library tab', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await page.getByRole('button', { name: 'Open Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
  })

  test('click digest Recent row opens Episode rail; stays on Digest', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await page
      .getByRole('button', {
        name: 'Digest Episode Alpha, Mock Feed Show',
        exact: true,
      })
      .click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(
      page
        .getByRole('region', { name: 'Episode', exact: true })
        .getByRole('heading', { name: 'Digest Episode Alpha' }),
    ).toBeVisible()
  })

  test('Digest ↔ Library keeps Episode rail when episode is in catalog', async ({
    page,
  }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await page
      .getByRole('button', {
        name: 'Digest Episode Alpha, Mock Feed Show',
        exact: true,
      })
      .click()
    const episodeRegion = page.getByRole('region', { name: 'Episode', exact: true })
    await expect(
      episodeRegion.getByRole('heading', { name: 'Digest Episode Alpha' }),
    ).toBeVisible()
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()
    await expect(
      episodeRegion.getByRole('heading', { name: 'Digest Episode Alpha' }),
    ).toBeVisible()
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await expect(
      episodeRegion.getByRole('heading', { name: 'Digest Episode Alpha' }),
    ).toBeVisible()
  })

  test('topic band title loads mocked GI and opens graph with canvas', async ({ page }) => {
    const ep1Gi = JSON.stringify({
      schema_version: '1.0',
      model_version: 'stub',
      prompt_version: 'v1',
      episode_id: 'e1',
      nodes: [
        {
          id: 'episode:e1',
          type: 'Episode',
          properties: {
            podcast_id: 'podcast:unknown',
            title: 'Digest Episode Alpha',
            publish_date: '2020-01-01T00:00:00Z',
          },
        },
      ],
      edges: [],
    })
    await page.route('**/api/artifacts/metadata/ep1.gi.json**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: ep1Gi,
      })
    })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await page.getByPlaceholder('/path/to/output').fill('/mock/corpus')
    await expect(page.getByTestId('digest-root')).toBeVisible()
    await page
      .getByRole('button', { name: 'Open top hit for topic Mock Topic Band in graph' })
      .click()
    await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
    await expect(page.locator('.graph-canvas')).toBeVisible()
  })
})
