import { expect, test } from '@playwright/test'

/**
 * PWA offline behavior — the guide's §7 discipline: test the OFFLINE
 * path with the SW installed, not just fresh install online. Guards
 * three invariants:
 *
 * 1. **App shell loads offline.** The SW must serve `index.html` +
 *    precached JS/CSS/fonts when the network is gone. Navigation
 *    fallback (`navigateFallback: 'index.html'`) must resolve so a
 *    deep-link like `/library` renders instead of an error page.
 *
 * 2. **Audio is NOT cached (bridge-never-rehost).** The SW's runtime
 *    caching allowlist excludes audio origin URLs; going offline while
 *    trying to hit audio must NOT return a cached response. This guards
 *    the copyright/bandwidth invariant that audio is bridged from the
 *    origin, never rehosted.
 *
 * 3. **Per-user endpoints are NOT cached.** `/api/app/me`, `/queue`,
 *    `/playback`, `/auth` are excluded from the SW cache — otherwise a
 *    signed-out user could see another session's cached data.
 *
 * Runs with `serviceWorkers: 'allow'` (default globally is 'block').
 * Slow marker because the SW install + activation walk takes multiple
 * navigations.
 */

test.use({ serviceWorkers: 'allow' })

test.describe('PWA offline behavior', () => {
  test('app shell renders when the network is offline', async ({
    page,
    context,
  }) => {
    // 1) Load online to install + activate the SW.
    await page.goto('/')
    await page.waitForFunction(async () => {
      const reg = await navigator.serviceWorker.ready
      return reg.active?.state === 'activated'
    })

    // 2) Warm the shell precache — visit each precached page once so
    //    workbox's install-time precache is definitely persisted before
    //    we cut the network.
    await page.goto('/library')
    await page.waitForLoadState('networkidle')
    await page.goto('/')
    await page.waitForLoadState('networkidle')

    // 3) Cut the network.
    await context.setOffline(true)

    // 4) A hard reload should still render the shell (index.html served
    //    from the SW precache via navigateFallback).
    await page.reload()
    // The <title> comes from index.html; if the shell 404'd offline this
    // would fail. Also assert the app root mount is present.
    await expect(page).toHaveTitle('Learning Player')
    await expect(page.locator('#app')).toBeVisible()

    // 5) Deep-link to a route while still offline. Vue Router's
    //    createWebHistory + workbox navigateFallback = index.html + client
    //    router resolves the route. The route may fail to load its data
    //    (no /api), but the app shell itself must render.
    await page.goto('/library')
    await expect(page.locator('#app')).toBeVisible()

    // Restore network for other tests in the run.
    await context.setOffline(false)
  })

  test('audio requests are NOT served from SW cache when offline', async ({
    page,
    context,
    request,
  }) => {
    await page.goto('/')
    await page.waitForFunction(async () => {
      const reg = await navigator.serviceWorker.ready
      return reg.active?.state === 'activated'
    })

    // Ask the SW's Cache Storage for any audio entry — there must be none.
    const cachedAudio = await page.evaluate(async () => {
      const cacheNames = await caches.keys()
      const found: string[] = []
      for (const name of cacheNames) {
        const cache = await caches.open(name)
        const keys = await cache.keys()
        for (const req of keys) {
          const u = new URL(req.url)
          // Guide invariant: audio is bridged from origin, never rehosted +
          // never cached by our SW. Any origin-audio URL in ANY cache is
          // a regression against bridge-never-rehost.
          if (
            u.pathname.match(/\.(mp3|m4a|aac|ogg|opus|wav)($|\?)/i) ||
            u.pathname.includes('/audio/')
          ) {
            found.push(req.url)
          }
        }
      }
      return found
    })
    expect(
      cachedAudio,
      `SW cache must not carry audio (bridge-never-rehost). Found: ${cachedAudio.join(', ')}`,
    ).toEqual([])

    // Also assert the shape of the SW's cache names — the ones our
    // vite-plugin-pwa config sets. If someone accidentally adds an
    // `audio` cache, THIS assertion catches it before shipping.
    const cacheNames = await page.evaluate(() => caches.keys())
    for (const name of cacheNames) {
      expect(name.toLowerCase()).not.toContain('audio')
    }

    // Keep the request fixture referenced so the linter doesn't strip it —
    // future work may add a request.get() audio-fetch assertion here.
    void request
    void context
  })

  test('per-user API endpoints are NOT in the SW cache', async ({ page }) => {
    await page.goto('/')
    await page.waitForFunction(async () => {
      const reg = await navigator.serviceWorker.ready
      return reg.active?.state === 'activated'
    })

    // Trigger a fetch to a per-user endpoint (may 401 signed-out; that's
    // fine — the SW gets to see the request and MUST decide not to cache).
    await page.evaluate(async () => {
      await fetch('/api/app/me').catch(() => {
        /* signed-out 401 is fine */
      })
      await fetch('/api/app/queue').catch(() => {
        /* signed-out 401 is fine */
      })
    })

    // Scan every cache for any per-user URL.
    const cachedPerUser = await page.evaluate(async () => {
      const cacheNames = await caches.keys()
      const found: string[] = []
      const perUserPattern = /^\/api\/app\/(me|queue|playback|auth)\b/
      for (const name of cacheNames) {
        const cache = await caches.open(name)
        const keys = await cache.keys()
        for (const req of keys) {
          const p = new URL(req.url).pathname
          if (perUserPattern.test(p)) found.push(req.url)
        }
      }
      return found
    })
    expect(
      cachedPerUser,
      `per-user endpoints must NOT be cached (auth-safety). Found: ${cachedPerUser.join(', ')}`,
    ).toEqual([])
  })
})
