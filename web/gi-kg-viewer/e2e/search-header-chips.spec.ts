import { expect, test, type Page, type Request } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-016 / UXS-005 Search header chips: each one relabels when set and changes what the NEXT
 * `/api/search` asks for (inspected from the request URL, as `search-operator-bar.spec.ts` does).
 * Min confidence is the exception by design — `/api/search` has no confidence param, so it
 * filters the current page client-side (`stores/search.ts` `filteredResults`).
 */

const QUERY = 'systems thinking'

async function openSearch(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
  await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
  await page.locator('#search-q').fill(QUERY)
}

/** Submit the query and return the `/api/search` request it issued. */
async function submit(page: Page): Promise<URL> {
  const req = page.waitForRequest(
    (r: Request) => new URL(r.url()).pathname === '/api/search' && r.method() === 'GET',
  )
  await page.locator('#search-q').press('Enter')
  return new URL((await req).url())
}

const results = (page: Page) => page.getByTestId('search-workspace').locator('article')

/** Type into a chip's popover input and close it with Enter (the chips' own close gesture). */
async function setChipText(page: Page, chip: string, value: string): Promise<void> {
  await page.getByTestId(`search-chip-${chip}`).click()
  const input = page.getByTestId(`search-popover-${chip}-input`)
  await input.fill(value)
  await input.press('Enter')
}

test.describe('Search header chips (UXS-016 / UXS-005)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
    await openSearch(page)
  })

  test('no chip set: the request carries none of the filter params', async ({ page }) => {
    const u = await submit(page)
    for (const p of ['topic', 'speaker', 'grounded_only', 'type']) {
      expect(u.searchParams.has(p), `unexpected ${p}=`).toBe(false)
    }
    expect(u.searchParams.get('top_k')).toBe('10')
  })

  test('Topic contains → "Topic: …" and topic=', async ({ page }) => {
    const chip = page.getByTestId('search-chip-topic-contains')
    await expect(chip).toHaveText('Topic ▾')
    await setChipText(page, 'topic-contains', 'governance')
    await expect(chip).toHaveText('Topic: governance ▾')
    expect((await submit(page)).searchParams.get('topic')).toBe('governance')
  })

  test('Speaker contains → "Speaker: …" and speaker=', async ({ page }) => {
    const chip = page.getByTestId('search-chip-speaker-contains')
    await expect(chip).toHaveText('Speaker ▾')
    await setChipText(page, 'speaker-contains', 'Maya')
    await expect(chip).toHaveText('Speaker: Maya ▾')
    expect((await submit(page)).searchParams.get('speaker')).toBe('Maya')
  })

  test('Grounded toggles to "Grounded ✓" and grounded_only=true', async ({ page }) => {
    const chip = page.getByTestId('search-chip-grounded-only')
    await expect(chip).toHaveText('Grounded')
    await chip.click()
    await expect(chip).toHaveText('Grounded ✓')
    await expect(chip).toHaveAttribute('aria-pressed', 'true')
    expect((await submit(page)).searchParams.get('grounded_only')).toBe('true')
  })

  test('Top-k → "Top‑k: 25" and top_k=25', async ({ page }) => {
    const chip = page.getByTestId('search-chip-topk')
    await expect(chip).toHaveText('Top‑k ▾')
    await setChipText(page, 'topk', '25')
    await expect(chip).toHaveText('Top‑k: 25 ▾')
    expect((await submit(page)).searchParams.get('top_k')).toBe('25')
  })

  test('Doc types → "Doc types: 2 of 6" and one type= per pick', async ({ page }) => {
    const chip = page.getByTestId('search-chip-doctypes')
    await expect(chip).toHaveText('Doc types ▾')
    await chip.click()
    const pop = page.getByTestId('search-popover-doctypes')
    await pop.getByRole('checkbox', { name: 'Insights' }).check()
    await pop.getByRole('checkbox', { name: 'Quotes' }).check()
    await expect(chip).toHaveText('Doc types: 2 of 6 ▾')
    await chip.click()
    const types = (await submit(page)).searchParams.getAll('type').sort()
    expect(types).toEqual(['insight', 'quote'])
  })

})

/**
 * Mocked so the page has known confidences (0.2 / 0.6 / 0.9 + one hit without any); the live
 * index's hits carry no stable confidence spread to threshold against.
 *
 * Fails on 2026-10-05 — app defect: the popover input is `<input type="number" v-model=…>`, and
 * Vue's v-model casts number inputs to a Number, while `SearchMinConfidenceChip.vue` (`isActive`,
 * `chipLabel`) and `stores/search.ts` (`filteredResults`) call `minConfidence.trim()`. Typing any
 * value throws `search.filters.minConfidence.trim is not a function` (page error, measured), the
 * chip never activates and nothing is filtered.
 */
test.describe('Search Min confidence chip (mocked page)', () => {
  test('"Min conf: 0.5" hides the hits below 0.5 without a new request; Clear restores them', async ({
    page,
  }) => {
    await mockSignIn(page, 'creator')
    await page.route('**/api/health**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', corpus_library_api: true, search_api: true }),
      }),
    )
    const hit = (id: string, confidence: number | null) => ({
      doc_id: `insight:${id}`,
      score: 0.8,
      text: `Insight ${id}`,
      metadata: {
        doc_type: 'insight',
        episode_id: `ep-${id}`,
        episode_title: `Episode ${id}`,
        feed_id: 'f1',
        ...(confidence == null ? {} : { confidence }),
      },
    })
    await page.route('**/api/search?**', (r) =>
      r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: QUERY,
          results: [hit('low', 0.2), hit('mid', 0.6), hit('high', 0.9), hit('none', null)],
          lift_stats: null,
          error: null,
          detail: null,
        }),
      }),
    )
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.locator('#search-q').fill(QUERY)
    await submit(page)
    await expect(results(page)).toHaveCount(4)

    let searchesAfter = 0
    page.on('request', (r) => {
      if (new URL(r.url()).pathname === '/api/search') searchesAfter += 1
    })
    const chip = page.getByTestId('search-chip-min-confidence')
    await expect(chip).toHaveText('Min conf ▾')
    await setChipText(page, 'min-confidence', '0.5')
    await expect(chip).toHaveText('Min conf: 0.5 ▾')
    await expect(results(page)).toHaveCount(2)
    await expect(page.getByTestId('search-workspace')).not.toContainText('Insight low')
    await expect(page.getByTestId('search-workspace')).not.toContainText('Insight none')

    await chip.click()
    await page.getByTestId('search-popover-min-confidence-clear').click()
    await expect(chip).toHaveText('Min conf ▾')
    await expect(results(page)).toHaveCount(4)
    expect(searchesAfter).toBe(0)
  })
})
