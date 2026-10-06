import { expect, test, type Page, type Response } from '@playwright/test'
import {
  dismissGraphGestureOverlayIfPresent,
  liveCorpusRoot,
  liveFeeds,
  mainViewsNav,
  mockSignIn,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * UXS-004 graph chrome on the live corpus: the time lens + counts strip (l.51–53), the search
 * highlight chip, and the #658 filter chip bar (l.162).
 *
 * The browser clock is pinned to 2026-07-17 for the lens test. The presets are relative to "now"
 * and the corpus is fixed, so an unpinned run drifts out of every window and the graph's
 * auto-widen (7d → 30d → 90d → all, App.vue `runCorpusGraphSyncBody`) silently turns every preset
 * into "all time". On 2026-07-17 the v3 corpus has 1 / 2 / 3 episodes inside 7 / 30 / 90 days.
 * (The config pins VITE_DEFAULT_GRAPH_LENS_DAYS=0, so the first load is "all time" — the 7-day
 * seed rule is deliberately not tested here.)
 */

const PINNED_NOW = new Date('2026-07-17T12:00:00Z')

async function openGraph(page: Page): Promise<void> {
  await page.goto('/')
  await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
  await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
  await mainViewsNav(page).getByRole('button', { name: 'Graph' }).click()
  await page.getByRole('button', { name: 'Fit' }).waitFor({ state: 'visible', timeout: 30_000 })
  await page.waitForLoadState('networkidle')
  await dismissGraphGestureOverlayIfPresent(page)
}

const isSiblingMerge = (r: Response): boolean =>
  r.url().endsWith('/api/corpus/resolve-episode-artifacts') && r.request().method() === 'POST'

/**
 * Apply a lens change and return the episode count once the graph has settled. The label flips at
 * once, but the reload — and the topic-cluster sibling merge that follows every reload
 * (`maybeMergeClusterSiblingEpisodes`, up to 10 extra episodes) — land later. Waiting for the
 * merge's POST is what makes the number deterministic; a quiet-network heuristic alone read the
 * count between the reload and the merge in 1 of 3 runs.
 */
async function lensChange(page: Page, act: () => Promise<void>): Promise<number> {
  const merged = page.waitForResponse(isSiblingMerge, { timeout: 30_000 })
  await act()
  await merged
  return stableEpisodeCount(page)
}

async function stableEpisodeCount(page: Page): Promise<number> {
  const count = page.getByTestId('graph-status-episode-count')
  let last = Number.NaN
  await expect
    .poll(
      async () => {
        await page.waitForLoadState('networkidle')
        const v = Number(await count.innerText())
        const stable = v === last
        last = v
        return stable
      },
      { intervals: [750], timeout: 20_000 },
    )
    .toBe(true)
  return last
}

test.describe('Graph time lens and counts strip (UXS-004)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('presets and the Since date relabel the strip and change the episode count', async ({
    page,
  }) => {
    await page.clock.setFixedTime(PINNED_NOW)
    await openGraph(page)
    const label = page.getByTestId('graph-status-lens-label')
    const lens = page.getByTestId('graph-status-lens-selector')
    await expect(label).toHaveText('Showing all time')
    await expect(page.getByTestId('graph-status-capped')).toBeVisible()

    const n7 = await lensChange(page, () =>
      lens.getByRole('button', { name: '7d', exact: true }).click(),
    )
    await expect(label).toHaveText('Showing last 7 days')

    const n30 = await lensChange(page, () =>
      lens.getByRole('button', { name: '30d', exact: true }).click(),
    )
    await expect(label).toHaveText('Showing last 30 days')

    const n90 = await lensChange(page, () =>
      lens.getByRole('button', { name: '90d', exact: true }).click(),
    )
    await expect(label).toHaveText('Showing last 90 days')

    // Each wider window holds one more corpus episode than the last (1 / 2 / 3).
    expect(n30).toBe(n7 + 1)
    expect(n90).toBe(n30 + 1)
    await expect(page.getByTestId('graph-status-capped')).toHaveCount(0)

    const since = page.getByTestId('graph-status-since-input')
    // `fill` already fires `change` on a date input; dispatching another would reload twice.
    const nSince = await lensChange(page, () => since.fill('2026-01-01'))
    await expect(label).toHaveText('Showing since 2026-01-01')
    // 2026-01-01 → 2026-07-17 holds 8 corpus episodes, five more than 90 days (p14_e03 on
    // 2026-01-01 and p10_e03 on 2026-01-19 are two of them).
    expect(nSince).toBe(n90 + 5)

    // All time is capped, so it can leave no sibling to merge — don't wait for one.
    await lens.getByRole('button', { name: 'All', exact: true }).click()
    await expect(label).toHaveText('Showing all time')
    await expect(page.getByTestId('graph-status-capped')).toBeVisible()
    await expect(page.getByTestId('graph-status-episode-count')).not.toHaveText(String(nSince))
    expect(await stableEpisodeCount(page)).toBeGreaterThan(nSince)
  })

  // The chip tracks active search highlights, not the Show-on-graph click itself (measured: going
  // to Graph by the nav after the search shows it too). Show on graph is the path a user takes.
  test('a search with hits puts the highlight chip above the canvas', async ({ page }) => {
    await openGraph(page)
    await expect(page.getByTestId('graph-search-highlight-chip')).toHaveCount(0)

    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.locator('#search-q').fill('risk management')
    await page.locator('#search-q').press('Enter')
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible({
      timeout: 30_000,
    })
    await page.getByRole('button', { name: 'Show on graph' }).first().click()

    await expect(page.getByTestId('graph-tab-panel')).toBeVisible()
    await expect(page.getByTestId('graph-search-highlight-chip')).toHaveText(
      /^\s*[1-9]\d*\s+highlights?\s*$/,
    )
  })
})

test.describe('Graph filter chip bar (UXS-004 #658)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator', { liveApi: true })
  })

  test('Types: Quote starts off; a change shows reset, which restores the defaults', async ({
    page,
  }) => {
    await openGraph(page)
    const chip = page.getByTestId('graph-chip-types')
    await expect(chip).toHaveText('Types ▾')
    await chip.click()
    const pop = page.getByTestId('graph-popover-types')
    await expect(pop).toBeVisible()
    const box = (name: string) => pop.getByRole('checkbox', { name: new RegExp(`^${name}\\b`) })
    // Neither the live corpus nor the offline fixture has Speaker nodes, so Quote carries the
    // default-off assertion. Episode is deliberately NOT asserted: UXS-004 says it starts off,
    // `utils/parsing.ts` DEFAULT_HIDDEN_TYPES leaves it on — a spec/code divergence to resolve,
    // not something to pin either way here.
    await expect(box('Quote')).not.toBeChecked()
    await expect(box('Insight')).toBeChecked()
    await expect(pop.getByTestId('graph-types-reset')).toHaveCount(0)

    await box('Insight').uncheck()
    await expect(chip).toHaveText(/^Types: \d+ of \d+ ▾$/)
    await pop.getByTestId('graph-types-reset').click()
    await expect(box('Insight')).toBeChecked()
    await expect(chip).toHaveText('Types ▾')
    await expect(pop.getByTestId('graph-types-reset')).toHaveCount(0)
  })

  test('Feed, Sources and Degree open their popovers, relabel when set, and reset all clears them', async ({
    page,
  }) => {
    await openGraph(page)
    const bar = page.getByTestId('graph-filter-bar')
    await expect(page.getByTestId('graph-chip-reset-all')).toHaveCount(0)

    // Feed
    const feedChip = page.getByTestId('graph-chip-feed')
    await expect(feedChip).toHaveText('Feed ▾')
    await feedChip.click()
    const feedPop = page.getByTestId('graph-popover-feed')
    await expect(feedPop).toBeVisible()
    const feed = (await liveFeeds(page))[0]!
    await feedPop.getByRole('button', { name: new RegExp(`^${feed.display_title},`) }).click()
    await expect(feedChip).toHaveText(`Feed: ${feed.display_title} ▾`)
    if (await feedPop.isVisible()) await feedChip.click()

    // Sources — the corpus graph is GI + KG, so the chip is offered.
    const sourcesChip = page.getByTestId('graph-chip-sources')
    await expect(sourcesChip).toHaveText('Sources ▾')
    await sourcesChip.click()
    const sourcesPop = page.getByTestId('graph-popover-sources')
    await expect(sourcesPop).toBeVisible()
    await sourcesPop.getByRole('checkbox', { name: 'KG' }).uncheck()
    await expect(sourcesChip).toHaveText('Sources: GI only ▾')
    await sourcesChip.click()

    // Degree
    const degreeChip = page.getByTestId('graph-chip-degree')
    await expect(degreeChip).toHaveText('Degree ▾')
    await degreeChip.click()
    const degreePop = page.getByTestId('graph-popover-degree')
    await expect(degreePop).toBeVisible()
    const bucket = degreePop.getByRole('button', { pressed: false }).first()
    const bucketId = ((await bucket.innerText()).split('(')[0] ?? '').trim()
    await bucket.click()
    await expect(degreeChip).toHaveText(`Degree: ${bucketId} ▾`)
    await degreeChip.click()

    const reset = bar.getByTestId('graph-chip-reset-all')
    await expect(reset).toBeVisible()
    await reset.click()
    await expect(feedChip).toHaveText('Feed ▾')
    await expect(sourcesChip).toHaveText('Sources ▾')
    await expect(degreeChip).toHaveText('Degree ▾')
    await expect(reset).toHaveCount(0)
  })
})
