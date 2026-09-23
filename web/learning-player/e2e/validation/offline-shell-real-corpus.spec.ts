import { expect, test } from '@playwright/test'

/**
 * Tier-3 — offline shell survives a REAL network drop.
 *
 * The fast-e2e `offline.spec.ts` runs against the preview build with a
 * committed synthetic corpus. This Tier-3 variant runs against
 * `make serve-for-validation` (production-shape backend + operator's
 * chosen corpus via `APP_CORPUS_PATH`) — catches SW-install / precache
 * / offline-fallback drift that only surfaces at production scale.
 */

test('operator offline: shell + deep-links survive network drop', async ({
  page,
  context,
}) => {
  await page.goto('/')
  await page.waitForFunction(async () => {
    const reg = await navigator.serviceWorker.ready
    return reg.active?.state === 'activated'
  })
  await page.screenshot({
    path: 'validation-results/offline-01-online.png',
    fullPage: true,
  })

  // Warm precache: touch several routes so their assets are in the SW cache
  // when we cut the network.
  await page.goto('/library')
  await page.waitForLoadState('networkidle')
  await page.goto('/search')
  await page.waitForLoadState('networkidle')
  await page.goto('/')
  await page.waitForLoadState('networkidle')

  // Cut network.
  await context.setOffline(true)

  // Reload — the shell (index.html + precached JS/CSS) must still render.
  await page.reload()
  // Regex, not an exact string, and it matches the fast `offline.spec.ts`.
  // Two different titles are BOTH correct here: `index.html` ships
  // "Close Listening", and once the SPA hydrates the router's `afterEach`
  // rewrites it to "Home · Close Listening" (`pageTitles.home` + `app.title`).
  // Asserting either exact value races the hydration; asserting the brand
  // covers both and still fails if the shell served nothing at all.
  await expect(page).toHaveTitle(/Close Listening/)
  await expect(page.locator('#app')).toBeVisible()
  await page.screenshot({
    path: 'validation-results/offline-02-shell-reload.png',
    fullPage: true,
  })

  // Deep-link to a route while offline — navigateFallback: index.html
  // routes it back to the shell, Vue Router resolves the target route.
  await page.goto('/library')
  await expect(page.locator('#app')).toBeVisible()
  await page.screenshot({
    path: 'validation-results/offline-03-deep-link.png',
    fullPage: true,
  })

  // Restore network so subsequent tests aren't affected.
  await context.setOffline(false)
})
