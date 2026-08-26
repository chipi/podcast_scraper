import { expect, test } from '@playwright/test'

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

// The coming-soon gate returns 200-HTML (not a 401 challenge), so Playwright's httpCredentials never
// fires for the `request` fixture — API calls would get the gate page. Send the preview basic-auth
// explicitly on API requests (the `page` fixture passes the gate via the /preview cookie instead).
const authHeaders: Record<string, string> = gated
  ? {
      Authorization:
        'Basic ' +
        Buffer.from(
          `${process.env.PLAYER_PREVIEW_USER || 'marko'}:${process.env.PLAYER_PREVIEW_PASS}`
        ).toString('base64'),
    }
  : {}

test.describe('public API contracts', () => {
  test.skip(!gated, 'set PLAYER_PREVIEW_PASS to run the gated live specs')

  test('core read endpoints return their expected shapes', async ({ request }) => {
    const episodes = await request.get('/api/app/episodes?page_size=1', { headers: authHeaders })
    expect(episodes.status()).toBe(200)
    const ep = await episodes.json()
    expect(Array.isArray(ep.items)).toBe(true)
    expect(ep.total).toBeGreaterThan(0)

    for (const path of ['/api/app/podcasts', '/api/app/theme-clusters?limit=3']) {
      const r = await request.get(path, { headers: authHeaders })
      expect(r.status(), path).toBe(200)
      expect(Array.isArray((await r.json()).items), path).toBe(true)
    }

    const search = await request.get('/api/app/search?q=ai&top_k=3', { headers: authHeaders })
    expect(search.status()).toBe(200)
    expect(Array.isArray((await search.json()).results)).toBe(true)
  })
})

test.describe('public UI surfaces', () => {
  test.skip(!gated, 'set PLAYER_PREVIEW_PASS to run the gated live specs')

  test('Home renders the hero + discovery tabs', async ({ page }) => {
    await page.goto('/preview')
    await expect(page.getByText("Find any moment you've heard.")).toBeVisible()
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
      await (await request.get('/api/app/episodes?page_size=15', { headers: authHeaders })).json()
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
    const feed = (await (await request.get('/api/app/podcasts', { headers: authHeaders })).json())
      .items?.[0]
    expect(feed?.feed_id, 'prod must have at least one show').toBeTruthy()
    await page.goto('/preview')
    await page.goto(`/podcast/${feed.feed_id}`)
    await expect(page).toHaveURL(/\/podcast\//)
    await expect(page.locator('main a[href^="/episode/"]').first()).toBeVisible()
  })
})
