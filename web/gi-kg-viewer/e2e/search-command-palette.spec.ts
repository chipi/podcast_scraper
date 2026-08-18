import { expect, test, type Page, type TestInfo } from '@playwright/test'
import {
  liveCorpusRoot,
  mainViewsNav,
  mockSignIn,
  resetUserPreferences,
  SHELL_HEADING_RE,
  signInIsolated,
  statusBarCorpusPathInput,
} from './helpers'

/**
 * Search v3 §S3 (#1233) — Cmd-K / `/` command palette Tier-1 contract.
 *
 * The palette is the shell's global search summon — summonable from any main tab, live-queries
 * ``/api/search`` (debounced 200 ms, top_k=8), and offers per-hit actions (Open in Workspace /
 * Pin to rail / Pin to Compare / Show on graph). Its default (query cleared) surface reads
 * USERPREFS-1 ``search.recentQueries`` and ``search.savedQueries``.
 *
 * Unit coverage (CommandPalette.test.ts): render + emit contracts. This Tier-1 spec covers the
 * live shell flow.
 *
 * ``/`` is the primary summon shortcut (Cmd-K/Ctrl-K optional; some headless-Firefox builds don't
 * fire meta shortcuts consistently).
 *
 * #1619 — migrated to the live index, except the no-results test (see its own describe below).
 * Queries here are real corpus queries; the palette's debounced fetch hits the real
 * ``/api/search``.
 */
const QUERY = 'systems thinking'

test.describe('Search — command palette (#1233)', () => {
  /** Sign in as a per-test identity with clean prefs, land on Digest, blur focus. */
  async function landOnDigestWithCorpus(
    page: Page,
    who: string,
    testInfo: TestInfo,
  ): Promise<void> {
    await signInIsolated(page, who, testInfo)
    await resetUserPreferences(page)
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    // Blur editable focus so `/` summons the palette (see useViewerKeyboard).
    await page.locator('body').click({ position: { x: 5, y: 5 } })
  }

  async function openPalette(page: Page): Promise<void> {
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(page.getByTestId('command-palette-input')).toBeFocused()
  }

  /** Type the query and wait for the palette's own debounced request to land. */
  async function queryPalette(page: Page, q = QUERY): Promise<void> {
    const searchRequest = page.waitForRequest((r) => {
      if (!r.url().includes('/api/search')) return false
      return new URL(r.url()).searchParams.get('q') === q
    })
    await page.getByTestId('command-palette-input').fill(q)
    await searchRequest
    await expect(page.getByTestId('command-palette-results')).toBeVisible({ timeout: 30_000 })
  }

  test('`/` opens the palette from Digest; input has autofocus', async ({ page }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-open', testInfo)
    await openPalette(page)
  })

  test('`/` opens the palette from every main tab that the shortcut respects', async ({
    page,
  }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-tabs', testInfo)
    // Library
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await openPalette(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    // Search — `/` should ALSO open the palette (there is no launcher after §S4-shell). Blur
    // first: the workspace renders ``#search-q``, which is editable-focused by default.
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await openPalette(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    // Digest again (round-trip; proves the shortcut still fires after tab switches).
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await openPalette(page)
  })

  test('empty-state renders Recent + Saved sections with honest empty copy', async ({
    page,
  }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-empty-state', testInfo)
    await openPalette(page)
    // Prefs were reset for this identity, so both sections show the empty-state paragraphs —
    // "empty" here means a clean user, not a missing backend.
    await expect(page.getByTestId('command-palette-recent-empty')).toBeVisible()
    await expect(page.getByTestId('command-palette-saved-empty')).toBeVisible()
  })

  test('debounced live query fires /api/search and renders the per-hit actions', async ({
    page,
  }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-query', testInfo)
    await openPalette(page)
    await queryPalette(page)
    // Results render with the per-hit actions (RFC-107 §S3).
    await expect(page.getByTestId('command-palette-action-open-workspace').first()).toBeVisible()
    await expect(page.getByTestId('command-palette-action-pin-rail').first()).toBeVisible()
    // Search v3 §S8 — "Pin to Compare" is the 4th per-hit action.
    await expect(page.getByTestId('command-palette-action-pin-compare').first()).toBeVisible()
    await expect(page.getByTestId('command-palette-action-show-graph').first()).toBeVisible()
  })

  test('"Pin to Compare" pins the hit subject and closes the palette (§S8)', async ({
    page,
  }, testInfo) => {
    // Tier-1 scope: asserts the palette action fires + closes. The downstream "pins prefill the
    // Compare picker slots" behaviour is covered at unit level (CompareOperatorPanel.test.ts),
    // because the Compare chip is gated on ≥ 2 in-hit subjects.
    await landOnDigestWithCorpus(page, 'palette-pin-compare', testInfo)
    await openPalette(page)
    await queryPalette(page)
    const pin = page.getByTestId('command-palette-action-pin-compare').first()
    await expect(pin).toBeVisible()
    await pin.click()
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
  })

  test('"Open in Workspace" switches main tab to Search and runs the query', async ({
    page,
  }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-open-workspace', testInfo)
    await openPalette(page)
    await queryPalette(page)
    await page.getByTestId('command-palette-action-open-workspace').first().click()
    // Palette closes; workspace mounts; query prefilled + run.
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('#search-q')).toHaveValue(QUERY)
    // Hit renders in the workspace.
    await expect(
      page.getByTestId('search-workspace').locator('article').first(),
    ).toBeVisible({ timeout: 30_000 })
  })

  test('"Show on graph" switches main tab to Graph and closes palette', async ({
    page,
  }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-show-graph', testInfo)
    await openPalette(page)
    await queryPalette(page)
    await page.getByTestId('command-palette-action-show-graph').first().click()
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible({ timeout: 10_000 })
  })

  test('Escape closes the palette', async ({ page }, testInfo) => {
    await landOnDigestWithCorpus(page, 'palette-escape', testInfo)
    await openPalette(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
  })
})

/**
 * #1619 — this one keeps its stub, and NOT because of the fixture.
 *
 * A dense vector index answers every query with its nearest neighbours, so there is no such thing
 * as a query that returns nothing. Verified against the live index: `q=zzzznomatch` came back with
 * **8 results**, top hit `kg_entity:…:person:sophie-laurent` at score 0.016. An empty result set is
 * therefore not reachable by asking — it has to be injected, exactly like a 404.
 *
 * No corpus version changes this; it is a property of semantic search. Treat as category C.
 */
test.describe('Search — command palette empty results (stubbed: unreachable via live search)', () => {
  test.beforeEach(async ({ page }) => {
    await mockSignIn(page, 'creator')
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
    await page.route('**/api/search?**', async (route) => {
      const q = new URL(route.request().url()).searchParams.get('q') ?? ''
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ query: q, query_type: 'semantic', results: [] }),
      })
    })
  })

  test('empty result set renders the "no results" line', async ({ page }) => {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await page.getByTestId('command-palette-input').fill('zzzznomatch')
    await expect(page.getByTestId('command-palette-no-results')).toBeVisible({ timeout: 5000 })
  })
})
