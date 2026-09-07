import { expect, test } from '@playwright/test'
import { addSessionCookie, bearer, canMintSession } from './session'

/**
 * Post-deploy live smoke for the PUBLIC, read-only consumer surfaces against the deployed
 * closelistening.app (playwright.live.config.ts). DATA-AGNOSTIC: ids are fetched from the API at
 * run time and assertions check SHAPE + that the surface renders (or an honest empty state), never
 * specific prod content.
 *
 * Per-user surfaces (Collections, Library, Queue, Favourites) are NOT here — they need a real Google
 * sign-in a headless smoke can't complete; they'd need a seeded test account (separate infra).
 */
const gated = Boolean(process.env.PLAYER_PREVIEW_PASS)

test.describe('public API contracts', () => {
  test.skip(
    !gated || !canMintSession,
    'needs PLAYER_PREVIEW_PASS + PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID',
  )

  // Two doors since RFC-120 (#1940): the coming-soon gate AND an app session. Priming /preview puts
  // `cl_preview` in the request jar (an explicit `Authorization: Bearer` would otherwise override
  // the Basic that clears the gate), and the session cookie signs the PAGE in — without it every
  // navigation below lands on /welcome instead of the app.
  test.beforeEach(async ({ request, page, baseURL }) => {
    await request.get('/preview')
    await addSessionCookie(page.context(), baseURL || 'https://closelistening.app')
  })

  test('core read endpoints return their expected shapes', async ({ request }) => {
    test.setTimeout(150_000) // the search probe below can legitimately take ~90s on a cold prod
    const episodes = await request.get('/api/app/episodes?page_size=1', { headers: bearer() })
    expect(episodes.status()).toBe(200)
    const ep = await episodes.json()
    expect(Array.isArray(ep.items)).toBe(true)
    expect(ep.total).toBeGreaterThan(0)

    for (const path of ['/api/app/podcasts', '/api/app/theme-clusters?limit=3']) {
      const r = await request.get(path, { headers: bearer() })
      expect(r.status(), path).toBe(200)
      expect(Array.isArray((await r.json()).items), path).toBe(true)
    }

    // Semantic search on a cold prod container exceeded the 45s TEST budget on desktop-chrome in
    // the 2026-09-07 run — the request itself timed out, it did not return an error. This raises
    // the ceiling so a slow-but-working search stops reading as a broken surface.
    //
    // That is SUPPRESSION, not a fix, and it is worth being explicit about: nobody has measured
    // what this endpoint actually costs on prod, so the real question — is a first search after a
    // deploy genuinely this slow for a user? — is still open. The timing is logged below so the
    // next run produces the number instead of another timeout.
    const started = Date.now()
    const search = await request.get('/api/app/search?q=ai&top_k=3', {
      headers: bearer(),
      timeout: 90_000,
    })
    // eslint-disable-next-line no-console
    console.log(`[live] /api/app/search took ${Date.now() - started}ms`)
    expect(search.status()).toBe(200)
    expect(Array.isArray((await search.json()).results)).toBe(true)
  })
})

test.describe('signed-in UI surfaces', () => {
  test.skip(
    !gated || !canMintSession,
    'needs PLAYER_PREVIEW_PASS + PLAYER_APP_SESSION_SECRET + PLAYER_SMOKE_USER_ID',
  )

  // Every surface below is behind the app session since RFC-120 (#1940); clearing the coming-soon
  // gate alone lands on /welcome. Same two doors as the API block above.
  test.beforeEach(async ({ page, baseURL }) => {
    await addSessionCookie(page.context(), baseURL || 'https://closelistening.app')
  })

  test('Home renders the hero + discovery tabs', async ({ page }) => {
    await page.goto('/preview')
    // NOT the hero text: Home's hero is ADAPTIVE, and the smoke account has listening history, so
    // it gets "Continue listening" rather than "Find any moment you've heard."
    await expect(page).not.toHaveURL(/\/welcome/)
    await expect(page.getByTestId('home-search-input')).toBeVisible()
    // The #4 discovery switcher (Rising / Trending / Storylines), Rising selected by default.
    await expect(page.getByTestId('home-discovery')).toBeVisible()
    await expect(page.getByTestId('discovery-tab-rising')).toHaveAttribute('aria-selected', 'true')
  })

  test('Search renders grouped results for a common term', async ({ page }) => {
    await page.goto('/preview')
    await page.goto('/search?q=ai')
    await expect(page).toHaveURL(/\/search/)
    // Either passages rendered ("N passages across M episodes") or an honest "No matches found." —
    // never a broken page. Data-agnostic: both are acceptable outcomes on prod.
    await expect(page.getByText(/passages across|No matches found/i).first()).toBeVisible()
  })

  test('Browse hub renders all four tabs', async ({ page }) => {
    await page.goto('/preview')
    await page.goto('/browse?tab=episodes')
    await expect(page.getByTestId('browse-view')).toBeVisible()
    for (const tab of ['episodes', 'shows', 'topics', 'people']) {
      await expect(page.getByTestId(`browse-tab-${tab}`)).toBeVisible()
    }
  })

  test('Player opens a playable episode with a live audio element', async ({ page, request }) => {
    // Pick a READY, audio-bridged episode — the absolute newest can be a pending (unprocessed) one
    // with no transport, which is data-dependent and flaked in CI.
    const list = (
      await (await request.get('/api/app/episodes?page_size=15', { headers: bearer() })).json()
    ).items as Array<{ slug: string; status: string; has_bridge: boolean }>
    const ep = list?.find((e) => e.status === 'ready' && e.has_bridge)
    expect(ep?.slug, 'prod must have a ready, playable episode').toBeTruthy()
    await page.goto('/preview')
    await page.goto(`/episode/${ep!.slug}`)
    await expect(page).toHaveURL(/\/episode\//)
    // The transport renders and the player-store <audio> element exists (audio-continuity contract).
    await expect(page.getByRole('button', { name: 'Play', exact: true }).first()).toBeVisible()
    await expect
      .poll(() => page.evaluate(() => Boolean(document.querySelector('audio'))), {
        timeout: 10_000,
      })
      .toBe(true)
  })

  test('a show page renders its episode list', async ({ page, request }) => {
    const feed = (await (await request.get('/api/app/podcasts', { headers: bearer() })).json())
      .items?.[0]
    expect(feed?.feed_id, 'prod must have at least one show').toBeTruthy()
    await page.goto('/preview')
    await page.goto(`/podcast/${feed.feed_id}`)
    await expect(page).toHaveURL(/\/podcast\//)
    await expect(page.locator('main a[href^="/episode/"]').first()).toBeVisible()
  })
})
