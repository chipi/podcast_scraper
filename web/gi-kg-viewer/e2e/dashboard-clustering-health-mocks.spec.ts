import { expect, test } from '@playwright/test'
import { setupCorpusDashboardDataRoutes } from './dashboardApiMocks'
import { openCorpusDataWorkspace, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * RFC-128 (#2349): the dashboard's "Storylines & themes" card — counts, registry health, the
 * chart, the last run's changes, and the history seed's preview → apply.
 *
 * Mocked, like the topic-clusters rebuild beside it: live, Apply writes a seed into the corpus
 * that the next storyline run takes in, and the history the card charts only exists once runs
 * have been recorded — the fixture corpus has none. The API is covered by the integration tests
 * (`test_corpus_clustering_health.py`), the seed by `test_storyline_history.py`.
 */
const HEALTH = {
  storylines: {
    latest: {
      ts: 1_791_936_000,
      source: 'artifact',
      ok: true,
      storylines: 99,
      active: 82,
      quiet: 17,
      candidates: 47,
      episodes_covered: 795,
      episodes_total: 2635,
      changes: { renamed: [{ id: 'thc:a', from: 'risk', to: 'risk management' }] },
    },
    health: { ok: true, source: 'artifact' },
    seed: { state: 'none', id: null },
    series: [
      { ts: 1_730_000_000, source: 'replay', storylines: 2, quiet: 0, candidates: 1 },
      { ts: 1_791_936_000, source: 'artifact', storylines: 99, quiet: 17, candidates: 47 },
    ],
  },
  themes: {
    latest: { ts: 1_791_936_000, themes: 2500, topics: 23166, kept: 2294, changes: {} },
    series: [
      { ts: 1_791_000_000, themes: 2442 },
      { ts: 1_791_936_000, themes: 2500 },
    ],
  },
}

test.describe('Storylines & themes card (mocked API)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'admin')
    await page.route('**/api/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, corpus_digest_api: true }),
      })
    })
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    await setupCorpusDashboardDataRoutes(page)
  })

  test('shows the counts and charts, then previews and applies the history seed', async ({ page }) => {
    let seed = 'none'
    await page.route('**/api/corpus/clustering/health**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...HEALTH,
          storylines: { ...HEALTH.storylines, seed: { state: seed, id: seed === 'none' ? null : 'abc' } },
        }),
      })
    })
    const modes: string[] = []
    await page.route('**/api/corpus/storylines/seed**', async (route) => {
      const mode = new URL(route.request().url()).searchParams.get('mode') ?? ''
      modes.push(mode)
      if (mode === 'apply') seed = 'pending'
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          mode === 'preview'
            ? { mode, current: 99, with_history: 51, folded: 59, added_from_history: [], candidates: 47, history_months: 24 }
            : { mode, applied: true },
        ),
      })
    })

    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await openCorpusDataWorkspace(page)
    await page.getByRole('tablist', { name: 'Dashboard tabs' }).getByRole('tab', { name: 'Intelligence' }).click()

    const card = page.getByTestId('clustering-health-card')
    await expect(card.getByTestId('clustering-storylines-count')).toContainText('99')
    await expect(card.getByTestId('clustering-registry-health')).toContainText('healthy')
    await expect(card.getByTestId('clustering-themes-count')).toContainText('2500')
    await expect(card.getByTestId('clustering-storyline-changes')).toContainText('risk → risk management')
    await expect(card.locator('canvas')).toHaveCount(2) // storylines and themes over time

    await card.getByTestId('clustering-seed-preview').click()
    await expect(card.getByTestId('clustering-seed-report')).toContainText('99 storylines today')
    await card.getByTestId('clustering-seed-apply').click()
    await expect(card.getByTestId('clustering-seed-state')).toHaveText('pending')
    await expect(card.getByTestId('clustering-seed-undo')).toBeVisible()
    expect(modes).toEqual(['preview', 'apply'])
  })
})
