import { expect, test } from '@playwright/test'
import {
  liveCorpusRoot,
  liveFirstEpisode,
  mainViewsNav,
  resetUserPreferences,
  SHELL_HEADING_RE,
  signInIsolated,
  statusBarCorpusPathInput,
} from '../helpers'

/**
 * Search v3 §S6 Tier-2 — rail-launched search walk.
 *
 * Library → Episode rail → "Search within episode" → scoped result set → clear chip → unscoped
 * fresh run. The cross-slice path: rail click → correctly-scoped `/api/search` → clear chip →
 * unscoped `/api/search`.
 *
 * #1619 — migrated to the live index.
 *
 * The old version stubbed `/api/search` to return an ep-alpha row only when `episode_id=ep-alpha`
 * was present, then asserted exactly one row rendered. That made "the server scopes retrieval"
 * true by construction. The live server scopes for real, so this now asserts the property itself:
 * every hit the scoped endpoint returns belongs to the episode, and dropping the scope widens the
 * result set beyond it.
 *
 * Scope note (unchanged): the sibling-scope-clear rule (§S6 resets feed / topic / speaker on rail
 * launch) is covered by the Tier-1 spec at `e2e/search-rail-in-episode.spec.ts`. Duplicating it
 * here would need a Pinia-devtools hook the viewer doesn't ship.
 */
const QUERY = 'systems thinking'

test.describe('Search v3 §S6 Tier-2 — rail-launched search walk', () => {
  test('rail launcher: Library → Episode → Search within episode → scoped result set → clear chip → unscoped fresh run', async ({
    page,
  }, testInfo) => {
    await signInIsolated(page, 'tier2-rail-launch', testInfo)
    await resetUserPreferences(page)

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
    await statusBarCorpusPathInput(page).fill(await liveCorpusRoot(page))
    await mainViewsNav(page).getByRole('button', { name: 'Library' }).click()
    await expect(page.getByTestId('library-root')).toBeVisible()

    const ep = await liveFirstEpisode(page)
    await page
      .getByRole('button', { name: `${ep.episode_title}, ${ep.feed_display_title}` })
      .click({ timeout: 15_000 })

    // ---- Trigger the rail launcher ----
    const railBtn = page.getByTestId('episode-detail-search-in-episode')
    await railBtn.waitFor({ state: 'visible', timeout: 15_000 })
    await railBtn.click()

    // ---- Search tab active + episode chip visible ----
    await expect(page.getByTestId('search-workspace')).toBeVisible({ timeout: 10_000 })
    const chip = page.getByTestId('search-chip-episode')
    await expect(chip).toBeVisible({ timeout: 10_000 })
    // The chip label is compact ("Episode ✕"); the episode_id lives on aria-label / title.
    await expect(chip).toHaveAttribute('aria-label', new RegExp(ep.episode_id))

    // ---- Scoped hit set renders ----
    const workspace = page.getByTestId('search-workspace')
    await expect(workspace.locator('article').first()).toBeVisible({ timeout: 30_000 })
    const scopedRendered = await workspace.locator('article').count()
    expect(scopedRendered).toBeGreaterThan(0)

    // ---- Verify episode_id landed on the wire ----
    await expect(async () => {
      const scoped = searchCalls.filter((c) => c.episodeId === ep.episode_id)
      if (!scoped.length) {
        throw new Error(`expected at least one /api/search with episode_id=${ep.episode_id}`)
      }
    }).toPass({ timeout: 5_000 })

    /* ---- The property the stub used to fake: the SERVER narrowed retrieval ---- */
    const scopedResp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&episode_id=${encodeURIComponent(ep.episode_id)}&top_k=10`,
    )
    const scopedBody = (await scopedResp.json()) as {
      results: { metadata?: { episode_id?: string } }[]
    }
    expect(scopedBody.results.length).toBeGreaterThan(0)
    expect(
      scopedBody.results.filter((r) => r.metadata?.episode_id !== ep.episode_id),
    ).toHaveLength(0)

    // ---- Clear the episode chip → scope removed from filter state ----
    // Chip click clears ``filters.episodeId`` but does NOT auto-re-run the search (shipped
    // behaviour — the user resubmits when they want a fresh page). Assert the scope removal plus
    // the unscoped wire format on resubmit.
    const callsBeforeClear = searchCalls.length
    await chip.click()
    await expect(chip).toHaveCount(0)
    await page.locator('#search-q').press('Enter')
    await expect(workspace.locator('article').first()).toBeVisible({ timeout: 30_000 })

    const postClear = searchCalls.slice(callsBeforeClear)
    const unscopedCalls = postClear.filter((c) => c.episodeId === null)
    expect(unscopedCalls.length).toBeGreaterThanOrEqual(1)

    /* Unscoped really is wider: the same query without the scope reaches beyond this episode. */
    const unscopedResp = await page.request.get(
      `/api/search?q=${encodeURIComponent(QUERY)}&top_k=10`,
    )
    const unscopedBody = (await unscopedResp.json()) as {
      results: { metadata?: { episode_id?: string } }[]
    }
    const otherEpisodes = new Set(
      unscopedBody.results
        .map((r) => r.metadata?.episode_id)
        .filter((id): id is string => Boolean(id) && id !== ep.episode_id),
    )
    expect(otherEpisodes.size).toBeGreaterThan(0)
  })
})
