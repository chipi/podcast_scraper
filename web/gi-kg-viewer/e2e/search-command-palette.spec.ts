import { expect, test } from '@playwright/test'
import { mainViewsNav, SHELL_HEADING_RE, statusBarCorpusPathInput, mockSignIn } from './helpers'

/**
 * Search v3 §S3 (#1233) — Cmd-K / `/` command palette Tier-1 contract.
 *
 * The palette is the shell's global search summon — summonable from any
 * main tab, live-queries ``/api/search`` (debounced 200 ms, top_k=8), and
 * offers three actions per hit (Open in Workspace / Pin to rail / Show
 * on graph). Its default (query cleared) surface reads USERPREFS-1
 * ``search.recentQueries`` for the Recent list and
 * ``search.savedQueries`` for the Saved list.
 *
 * Unit coverage (CommandPalette.test.ts): render + emit contracts for
 * the mounted component. This Tier-1 spec covers the live shell flow:
 * summoning from a real page, keyboard focus contract, debounced fetch,
 * hit-row 3-action rendering, and the "Open in Workspace" handoff.
 *
 * ``/`` is the primary summon shortcut (Cmd-K/Ctrl-K optional; some
 * headless-Firefox builds don't fire meta shortcuts consistently).
 */
test.describe('Search — command palette (#1233)', () => {
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
    await page.route('**/api/app/preferences', async (route) => {
      const method = route.request().method()
      if (method === 'GET') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences: {} }),
        })
        return
      }
      if (method === 'PATCH') {
        const req = route.request().postDataJSON() as
          | { preferences?: Record<string, unknown> }
          | null
        const preferences = req?.preferences ?? {}
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences }),
        })
        return
      }
      await route.fulfill({ status: 405, body: '' })
    })
    // Search route: one static hit for "climate" queries + empty for
    // everything else so the palette's live-fetch is deterministic.
    await page.route('**/api/search?**', async (route) => {
      const url = new URL(route.request().url())
      const q = url.searchParams.get('q')?.toLowerCase() ?? ''
      const results = q.includes('climate')
        ? [
            {
              doc_id: 'insight:e1:n1',
              score: 0.91,
              source_tier: 'insight',
              text: 'An insight about climate policy.',
              metadata: {
                doc_type: 'insight',
                episode_id: 'e-climate',
                source_id: 'insight:e1:n1',
                publish_date: '2026-04-15',
              },
            },
          ]
        : []
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: q,
          query_type: 'semantic',
          results,
        }),
      })
    })
  })

  async function landOnDigestWithCorpus(page: import('@playwright/test').Page) {
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    // Blur editable focus so `/` summons the palette (see useViewerKeyboard).
    await page.locator('body').click({ position: { x: 5, y: 5 } })
  }

  async function openPalette(page: import('@playwright/test').Page) {
    await page.keyboard.press('/')
    await expect(page.getByTestId('command-palette')).toBeVisible()
    await expect(page.getByTestId('command-palette-input')).toBeFocused()
  }

  test('`/` opens the palette from Digest; input has autofocus', async ({ page }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
  })

  test('`/` opens the palette from every main tab that the shortcut respects', async ({
    page,
  }) => {
    await landOnDigestWithCorpus(page)
    // Library
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await openPalette(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    // Search
    await mainViewsNav(page).getByRole('button', { name: 'Search' }).click()
    // On Search, `/` should ALSO open the palette (not focus the launcher —
    // there is no launcher after §S4-shell). Blur first — the workspace
    // tab renders ``#search-q`` which is editable-focused by default.
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await openPalette(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    // Digest again (round-trip; also proves the shortcut still fires after a
    // sequence of tab switches).
    await mainViewsNav(page).getByRole('button', { name: 'Digest' }).click()
    await page.locator('body').click({ position: { x: 5, y: 5 } })
    await openPalette(page)
  })

  test('empty-state renders Recent + Saved sections with honest empty copy', async ({ page }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
    // With no writes yet, both sections show the empty-state paragraphs.
    await expect(page.getByTestId('command-palette-recent-empty')).toBeVisible()
    await expect(page.getByTestId('command-palette-saved-empty')).toBeVisible()
  })

  test('debounced live query fires /api/search and renders three-action rows per hit', async ({
    page,
  }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
    const searchRequest = page.waitForRequest((r) => {
      if (!r.url().includes('/api/search')) return false
      const u = new URL(r.url())
      return u.searchParams.get('q') === 'climate'
    })
    await page.getByTestId('command-palette-input').fill('climate')
    // Debounce is 200 ms in CommandPalette.vue — allow ≥ 300 ms.
    await searchRequest
    // Results render with all three action buttons (RFC-107 §S3).
    await expect(page.getByTestId('command-palette-results')).toBeVisible()
    await expect(page.getByTestId('command-palette-action-open-workspace')).toBeVisible()
    await expect(page.getByTestId('command-palette-action-pin-rail')).toBeVisible()
    // Show on graph resolves via the hit's graph id; the mock's insight hit
    // has ``source_id`` so it should be enabled.
    await expect(page.getByTestId('command-palette-action-show-graph')).toBeVisible()
  })

  test('empty result set renders the "no results" line', async ({ page }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
    await page.getByTestId('command-palette-input').fill('zzzznomatch')
    await expect(page.getByTestId('command-palette-no-results')).toBeVisible({ timeout: 5000 })
  })

  test('"Open in Workspace" switches main tab to Search and runs the query', async ({ page }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
    await page.getByTestId('command-palette-input').fill('climate')
    await expect(page.getByTestId('command-palette-action-open-workspace')).toBeVisible()
    await page.getByTestId('command-palette-action-open-workspace').click()
    // Palette closes; workspace mounts; query prefilled + run.
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('#search-q')).toHaveValue('climate')
    // Hit renders in the workspace.
    await expect(page.getByTestId('search-workspace').locator('article').first()).toBeVisible()
  })

  test('"Show on graph" switches main tab to Graph and closes palette', async ({ page }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
    await page.getByTestId('command-palette-input').fill('climate')
    await expect(page.getByTestId('command-palette-action-show-graph').first()).toBeVisible()
    await page.getByTestId('command-palette-action-show-graph').first().click()
    // Palette closes; Graph tab-panel mounts (mocked corpus has no graph
    // artifacts → empty state OK).
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
    await expect(page.getByTestId('graph-tab-panel')).toBeVisible({ timeout: 10_000 })
  })

  test('Escape closes the palette', async ({ page }) => {
    await landOnDigestWithCorpus(page)
    await openPalette(page)
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('command-palette')).toHaveCount(0)
  })
})
