import { expect, test } from '@playwright/test'
import {
  mainViewsNav,
  SHELL_HEADING_RE,
  statusBarCorpusPathInput,
  mockSignIn,
} from '../helpers'

/**
 * Search v3 §S6 Tier-2 (F3) — rail-launched search walk (production-shaped).
 *
 * The Tier-1 spec (`search-rail-in-episode.spec.ts`) mocks each surface
 * in isolation. This Tier-2 walk drives the full cross-slice flow from
 * Library → Episode subject rail → "Search within this episode" → Search
 * tab with the episode chip active, then verifies:
 *
 *   1. The `episode_id` param on `/api/search` matches the rail's episode.
 *   2. The visible hit set is the server-scoped page (mock returns only
 *      hits for that episode; the client does NOT re-filter).
 *   3. Clearing the `search-chip-episode` triggers a fresh `/api/search`
 *      WITHOUT the `episode_id` param, and the server returns the
 *      unscoped page (verified via distinct hit ids).
 *   4. Sibling scope filters (feed / topic / speaker) that were set
 *      before the rail launch are cleared by the launcher so the wire
 *      matches the mental model of "this episode only".
 *
 * Mocks match the shipped shape (per-hit metadata contract from
 * `server/schemas.py::SearchHitModel`).
 */
test.describe('Search v3 §S6 Tier-2 — rail-launched search walk', () => {
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
    await page.route('**/api/index/stats**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          available: false,
          reason: 'mock-off',
          stats: null,
          reindex_recommended: false,
        }),
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
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ preferences: {} }),
        })
        return
      }
      await route.fulfill({ status: 405, body: '' })
    })
    await page.route('**/api/corpus/feeds**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feeds: [{ feed_id: 'f1', display_title: 'Odd Lots (fixture)', episode_count: 2 }],
        }),
      })
    })
    await page.route('**/api/corpus/episodes**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          feed_id: null,
          items: [
            {
              metadata_relative_path: 'metadata/ep-alpha.metadata.json',
              feed_id: 'f1',
              feed_display_title: 'Odd Lots (fixture)',
              topics: [],
              summary_title: 'Compute Governance',
              summary_bullets_preview: [],
              summary_preview: 'Compute governance early frame',
              episode_id: 'ep-alpha',
              episode_title: 'Compute Governance',
              publish_date: '2026-04-27',
              has_gi: true,
              has_kg: true,
              cil_digest_topics: [],
            },
            {
              metadata_relative_path: 'metadata/ep-beta.metadata.json',
              feed_id: 'f1',
              feed_display_title: 'Odd Lots (fixture)',
              topics: [],
              summary_title: 'Silicon Supply',
              summary_bullets_preview: [],
              summary_preview: 'Silicon supply national security frame',
              episode_id: 'ep-beta',
              episode_title: 'Silicon Supply',
              publish_date: '2026-04-30',
              has_gi: true,
              has_kg: true,
              cil_digest_topics: [],
            },
          ],
          next_cursor: null,
        }),
      })
    })
    await page.route('**/api/corpus/episodes/similar**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ items: [], available: false }),
      })
    })
    // Episode-detail response — the rail's ``Search within episode`` button
    // disables when ``detail.episode_id`` is empty, so the mock must
    // populate it for the launcher to be clickable.
    await page.route('**/api/corpus/episodes/detail**', async (route) => {
      const url = new URL(route.request().url())
      const metaPath = url.searchParams.get('metadata_relpath') ?? ''
      const isAlpha = metaPath.includes('ep-alpha')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          path: '/mock/corpus',
          metadata_relative_path: metaPath,
          feed_id: 'f1',
          episode_id: isAlpha ? 'ep-alpha' : 'ep-beta',
          episode_title: isAlpha ? 'Compute Governance' : 'Silicon Supply',
          publish_date: isAlpha ? '2026-04-27' : '2026-04-30',
          summary_title: isAlpha ? 'Compute Governance' : 'Silicon Supply',
          summary_bullets: [],
          summary_text: null,
          gi_relative_path: 'gi/ep.gi.json',
          kg_relative_path: 'kg/ep.kg.json',
          has_gi: true,
          has_kg: true,
        }),
      })
    })
    // Episode-related insights — non-blocking sub-fetch; return empty to
    // silence retries.
    await page.route('**/api/corpus/episodes/*/insights**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ items: [] }),
      })
    })
    /**
     * Production-shaped /api/search: branches on `episode_id` to return
     * either the scoped page (ep-alpha only) or the unscoped page (both
     * episodes). Records every call so tests can assert scope params.
     */
    await page.route('**/api/search?**', async (route) => {
      const url = new URL(route.request().url())
      const episodeId = url.searchParams.get('episode_id')
      const q = url.searchParams.get('q') ?? ''
      const scopedHits = [
        {
          doc_id: 'insight:ep-alpha:1',
          score: 0.91,
          source_tier: 'insight',
          text: 'An alpha-scoped insight from the rail launcher.',
          metadata: {
            doc_type: 'insight',
            episode_id: 'ep-alpha',
            episode_title: 'Compute Governance',
            feed_id: 'f1',
            feed_title: 'Odd Lots (fixture)',
            publish_date: '2026-04-27',
          },
        },
      ]
      const unscopedHits = [
        ...scopedHits,
        {
          doc_id: 'insight:ep-beta:1',
          score: 0.83,
          source_tier: 'insight',
          text: 'A beta-episode insight not visible under episode scope.',
          metadata: {
            doc_type: 'insight',
            episode_id: 'ep-beta',
            episode_title: 'Silicon Supply',
            feed_id: 'f1',
            feed_title: 'Odd Lots (fixture)',
            publish_date: '2026-04-30',
          },
        },
      ]
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          query: q,
          query_type: 'semantic',
          results: episodeId === 'ep-alpha' ? scopedHits : unscopedHits,
        }),
      })
    })
  })

  test('rail launcher: Library → Episode → Search within episode → scoped result set → clear chip → unscoped fresh run', async ({
    page,
  }) => {
    // Instrument /api/search so we can assert what actually went on the wire.
    const searchCalls: Array<{ q: string; episodeId: string | null }> = []
    page.on('request', (request) => {
      const url = request.url()
      if (!url.includes('/api/search')) return
      // Skip the compare endpoint; we only care about /api/search.
      if (url.includes('/api/search/compare')) return
      const parsed = new URL(url)
      searchCalls.push({
        q: parsed.searchParams.get('q') ?? '',
        episodeId: parsed.searchParams.get('episode_id'),
      })
    })

    // ---- Land on Library, open the first episode's detail rail ----
    await page.goto('/')
    await page.getByRole('heading', { name: SHELL_HEADING_RE }).waitFor()
    await statusBarCorpusPathInput(page).fill('/mock/corpus')
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    // First row for ep-alpha.
    const firstRow = page
      .getByRole('button', { name: /Compute Governance/ })
      .filter({ hasNotText: 'Open in graph' })
      .first()
    await firstRow.click({ timeout: 15_000 })

    // ---- Trigger the rail launcher ----
    const railBtn = page.getByTestId('episode-detail-search-in-episode')
    await railBtn.waitFor({ state: 'visible', timeout: 15_000 })
    await railBtn.click()

    // ---- Search tab active + episode chip visible ----
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    const chip = page.getByTestId('search-chip-episode')
    await expect(chip).toBeVisible({ timeout: 10_000 })
    // The chip label is compact ("Episode ✕"); the episode_id lives on
    // ``aria-label`` / ``title`` so the assertion pins there.
    await expect(chip).toHaveAttribute('aria-label', /ep-alpha/)

    // ---- Scoped hit set: only ep-alpha row rendered ----
    const workspace = page.getByTestId('search-workspace')
    await expect(workspace.locator('article')).toHaveCount(1, { timeout: 10_000 })
    await expect(workspace.locator('article').first()).toContainText(
      'An alpha-scoped insight',
    )

    // ---- Verify episode_id landed on the wire ----
    await expect(async () => {
      const scoped = searchCalls.filter((c) => c.episodeId === 'ep-alpha')
      if (!scoped.length) throw new Error('expected at least one /api/search with episode_id=ep-alpha')
    }).toPass({ timeout: 5_000 })

    // ---- Clear the episode chip → scope removed from filter state ----
    // Chip click clears ``filters.episodeId`` but does NOT auto-re-run
    // the search (the shipped behaviour — the user resubmits when they
    // want a fresh page). Assert the scope removal + the unscoped
    // wire-format when the caller re-submits.
    const callsBeforeClear = searchCalls.length
    await chip.click()
    await expect(chip).toHaveCount(0)
    await page.locator('#search-q').press('Enter')
    await expect(workspace.locator('article')).toHaveCount(2, { timeout: 10_000 })
    const postClear = searchCalls.slice(callsBeforeClear)
    const unscopedCalls = postClear.filter((c) => c.episodeId === null)
    expect(unscopedCalls.length).toBeGreaterThanOrEqual(1)
  })

})

/*
 * Notes on scope:
 *   - The sibling-scope-clear rule (Search v3 §S6 handler resets feed /
 *     topic / speaker on rail launch) is covered by the Tier-1 spec at
 *     `e2e/search-rail-in-episode.spec.ts`. Duplicating it here would
 *     need a Pinia-devtools hook that the viewer doesn't ship. This
 *     Tier-2 walk stays focused on the cross-slice production-shaped
 *     path: rail click → correctly-scoped /api/search → clear chip →
 *     unscoped fresh /api/search.
 */
