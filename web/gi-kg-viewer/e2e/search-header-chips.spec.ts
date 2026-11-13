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
 * The Grounded and Min confidence chips are a downstream build's (E04 step 8b): not offered here.
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

  test('Grounded and Min confidence are not offered', async ({ page }) => {
    await expect(page.getByTestId('search-chip-grounded-only')).toHaveCount(0)
    await expect(page.getByTestId('search-chip-min-confidence')).toHaveCount(0)
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

  test('Top-k → "Top‑k: 25" and top_k=25', async ({ page }) => {
    const chip = page.getByTestId('search-chip-topk')
    await expect(chip).toHaveText('Top‑k ▾')
    await setChipText(page, 'topk', '25')
    await expect(chip).toHaveText('Top‑k: 25 ▾')
    expect((await submit(page)).searchParams.get('top_k')).toBe('25')
  })

  test('Doc types → "Doc types: 2 of 4" and one type= per pick', async ({ page }) => {
    const chip = page.getByTestId('search-chip-doctypes')
    await expect(chip).toHaveText('Doc types ▾')
    await chip.click()
    const pop = page.getByTestId('search-popover-doctypes')
    // Insights and Quotes are a downstream build's doc types (E04 step 8b).
    await expect(pop.getByRole('checkbox', { name: 'Insights' })).toHaveCount(0)
    await expect(pop.getByRole('checkbox', { name: 'Quotes' })).toHaveCount(0)
    await pop.getByRole('checkbox', { name: 'KG entities' }).check()
    await pop.getByRole('checkbox', { name: 'KG topics' }).check()
    await expect(chip).toHaveText('Doc types: 2 of 4 ▾')
    await chip.click()
    const types = (await submit(page)).searchParams.getAll('type').sort()
    expect(types).toEqual(['kg_entity', 'kg_topic'])
  })

})
