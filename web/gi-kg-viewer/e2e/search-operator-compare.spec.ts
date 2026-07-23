import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  mockSignIn,
} from './helpers'

/**
 * Search v3 §S8 — ResultSetOperatorBar Compare chip + CompareOperatorPanel
 * end-to-end contract (mocked). Covers:
 *   - Compare chip enabled when ≥ 2 comparable subjects appear in the
 *     current hit set; disabled otherwise.
 *   - Toggling the chip opens the picker panel, seeded from the
 *     visible-hit subject discovery walk.
 *   - "Run compare" posts to /api/search/compare and renders both packs
 *     side-by-side.
 *   - The judge summary renders when both packs are grounded, and is
 *     absent when one side reports grounded=false.
 *   - Compare error state surfaces via operator-compare-error.
 *
 * The E2E surface map — [E2E_SURFACE_MAP.md](E2E_SURFACE_MAP.md) — is the
 * canonical selector contract; the new testids referenced here are
 * documented under the "Search v3 Compare (§S8)" block of that file.
 */
test.describe('Search — Compare operator (§S8)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
  })

  test.beforeEach(async ({ page }) => {
    await page.route('**/api/health**', async (route) => {
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
    await page.route('**/api/artifacts?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: '/mock/corpus', artifacts: [] }),
      })
    })
    // Plain /api/search — returns 2 hits from 2 distinct persons + topic +
    // episode metadata so the discovery walk finds ≥ 2 subjects.
    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'compute',
          query_type: 'semantic',
          results: [
            {
              doc_id: 'insight:e1:1',
              score: 0.9,
              source_tier: 'insight',
              text: 'A insight about compute',
              metadata: {
                doc_type: 'insight',
                speaker_name: 'Alice',
                episode_id: 'ep1',
                episode_title: 'Ep One',
                topic_label: 'Compute',
              },
            },
            {
              doc_id: 'insight:e2:1',
              score: 0.7,
              source_tier: 'insight',
              text: 'B insight about compute',
              metadata: {
                doc_type: 'insight',
                speaker_name: 'Bob',
                episode_id: 'ep2',
                episode_title: 'Ep Two',
                topic_label: 'Compute',
              },
            },
          ],
        }),
      })
    })
  })

  async function landOnSearchAndQuery(page: import('@playwright/test').Page) {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.locator('#search-q').fill('compute')
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article')).toHaveCount(2, {
      timeout: 10_000,
    })
  }

  test('the Compare chip enables when ≥ 2 comparable subjects appear in the hit set', async ({
    page,
  }) => {
    await landOnSearchAndQuery(page)
    const chip = page.getByTestId('operator-chip-compare')
    await expect(chip).toBeVisible()
    await expect(chip).toBeEnabled()
    // Toggle open — panel + picker render.
    await chip.click()
    await expect(page.getByTestId('operator-compare-panel')).toBeVisible()
    await expect(page.getByTestId('operator-compare-picker')).toBeVisible()
  })

  test('running Compare posts to /api/search/compare and renders 2 grounded packs + judge summary', async ({
    page,
  }) => {
    await page.route('**/api/search/compare', async (route) => {
      const body = route.request().postDataJSON() as {
        subject_a: { id: string; label: string; kind: string }
        subject_b: { id: string; label: string; kind: string }
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          pack_a: {
            subject: body.subject_a,
            query: 'compute',
            query_type: 'semantic',
            rendered: '[CRITICAL] A',
            token_count: 8,
            max_tokens: 2000,
            top_insight_id: 'insight:e1:1',
            top_insight_text: 'A insight about compute',
            supporting_segment_ids: [],
            supporting_segment_texts: [],
            coverage_summary: { episode_count: 3, show_ids: [], date_range: null },
            confidence_p50: 0.9,
            result_count: 3,
            grounded: true,
          },
          pack_b: {
            subject: body.subject_b,
            query: 'compute',
            query_type: 'semantic',
            rendered: '[CRITICAL] B',
            token_count: 6,
            max_tokens: 2000,
            top_insight_id: 'insight:e2:1',
            top_insight_text: 'B insight about compute',
            supporting_segment_ids: [],
            supporting_segment_texts: [],
            coverage_summary: { episode_count: 1, show_ids: [], date_range: null },
            confidence_p50: 0.5,
            result_count: 1,
            grounded: true,
          },
          judge_summary: `${body.subject_a.label} shows higher confidence (0.90 vs 0.50 for ${body.subject_b.label})`,
          error: null,
          detail: null,
        }),
      })
    })

    await landOnSearchAndQuery(page)
    await page.getByTestId('operator-chip-compare').click()
    await page.getByTestId('operator-compare-run').click()
    await expect(page.getByTestId('operator-compare-columns')).toBeVisible()
    await expect(page.getByTestId('operator-compare-pack-a')).toContainText('A insight about compute')
    await expect(page.getByTestId('operator-compare-pack-b')).toContainText('B insight about compute')
    await expect(page.getByTestId('operator-compare-judge')).toContainText('shows higher confidence')
    // Neither ungrounded badge should appear for two grounded packs.
    await expect(page.getByTestId('operator-compare-pack-a-ungrounded')).toHaveCount(0)
    await expect(page.getByTestId('operator-compare-pack-b-ungrounded')).toHaveCount(0)
  })

  test('the judge summary is muted when one side reports grounded=false', async ({
    page,
  }) => {
    await page.route('**/api/search/compare', async (route) => {
      const body = route.request().postDataJSON() as {
        subject_a: { id: string; label: string; kind: string }
        subject_b: { id: string; label: string; kind: string }
      }
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          pack_a: {
            subject: body.subject_a,
            query: 'compute',
            query_type: 'semantic',
            rendered: '',
            token_count: 0,
            max_tokens: 2000,
            top_insight_id: 'insight:e1:1',
            top_insight_text: 'A insight',
            supporting_segment_ids: [],
            supporting_segment_texts: [],
            coverage_summary: { episode_count: 3 },
            confidence_p50: 0.9,
            result_count: 3,
            grounded: true,
          },
          pack_b: {
            subject: body.subject_b,
            query: 'compute',
            query_type: 'semantic',
            rendered: '',
            token_count: 0,
            max_tokens: 2000,
            top_insight_id: null,
            top_insight_text: '',
            supporting_segment_ids: [],
            supporting_segment_texts: [],
            coverage_summary: {},
            confidence_p50: 0.0,
            result_count: 0,
            grounded: false,
          },
          judge_summary: null,
          error: null,
          detail: null,
        }),
      })
    })

    await landOnSearchAndQuery(page)
    await page.getByTestId('operator-chip-compare').click()
    await page.getByTestId('operator-compare-run').click()
    await expect(page.getByTestId('operator-compare-columns')).toBeVisible()
    await expect(page.getByTestId('operator-compare-judge')).toHaveCount(0)
    await expect(page.getByTestId('operator-compare-pack-b-ungrounded')).toBeVisible()
  })

  test('the Compare chip is disabled when the hit set has < 2 comparable subjects', async ({
    page,
  }) => {
    // Override /api/search to return a single hit → only 1 person / topic /
    // episode discoverable → chip disabled.
    await page.unroute('**/api/search?**')
    await page.route('**/api/search?**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: 'compute',
          query_type: 'semantic',
          results: [
            {
              doc_id: 'insight:e1:1',
              score: 0.9,
              source_tier: 'insight',
              text: 'A insight',
              metadata: { doc_type: 'insight', speaker_name: 'Alice' },
            },
          ],
        }),
      })
    })
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.locator('#search-q').fill('compute')
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article')).toHaveCount(1, {
      timeout: 10_000,
    })
    const chip = page.getByTestId('operator-chip-compare')
    await expect(chip).toBeVisible()
    await expect(chip).toBeDisabled()
  })

  test('server error surfaces via operator-compare-error and clears prior result', async ({
    page,
  }) => {
    await page.route('**/api/search/compare', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'text/plain',
        body: 'boom',
      })
    })
    await landOnSearchAndQuery(page)
    await page.getByTestId('operator-chip-compare').click()
    await page.getByTestId('operator-compare-run').click()
    await expect(page.getByTestId('operator-compare-error')).toContainText('boom')
    await expect(page.getByTestId('operator-compare-columns')).toHaveCount(0)
  })
})
