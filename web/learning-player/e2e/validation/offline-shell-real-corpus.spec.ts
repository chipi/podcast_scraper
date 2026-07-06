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
  await expect(page).toHaveTitle('Learning Player')
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
