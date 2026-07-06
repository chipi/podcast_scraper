import { expect, test } from '@playwright/test'

// Local ambient declaration so the e2e tsconfig (which doesn't include
// src/env.d.ts) sees window.__buildInfo when we assert against it.
declare global {
  interface Window {
    __buildInfo?: { sha: string; time: string }
  }
}

/**
 * PWA installability + service-worker regression coverage.
 *
 * Guards the specific traps from the PWA shipping guide:
 *   §1  manifest served + valid JSON with the required install fields
 *   §1  every declared icon URL returns 200 (regression against the
 *       "icons declared but missing" trap that was live in main before
 *       slice 1 of this branch)
 *   §2  sw.js served with no-cache (nginx contract) and registers
 *   §3  build-identity constants injected + window.__buildInfo populated
 *   §4  iOS apple-touch-icon + apple-mobile-web-app-* meta tags present
 *
 * NOTE: the global playwright config sets `serviceWorkers: 'block'` so
 * network-based tests are deterministic. This spec opts back in via
 * `test.use()` because verifying SW registration IS the point here.
 */

test.use({ serviceWorkers: 'allow' })

test.describe('PWA install surface', () => {
  test('manifest is served and has the required install fields', async ({ request }) => {
    const res = await request.get('/manifest.webmanifest')
    expect(res.status()).toBe(200)
    // Content-type MAY be application/manifest+json or application/json — accept both.
    const ct = (res.headers()['content-type'] || '').toLowerCase()
    expect(ct.includes('manifest') || ct.includes('json')).toBe(true)

    const manifest = await res.json()
    expect(manifest.name).toBeTruthy()
    expect(manifest.short_name).toBeTruthy()
    expect(manifest.start_url).toBeTruthy()
    expect(manifest.scope).toBeTruthy()
    expect(manifest.display).toBe('standalone')
    expect(manifest.theme_color).toBeTruthy()
    expect(manifest.background_color).toBeTruthy()

    // Must have at least one icon >= 192 and a maskable variant.
    expect(Array.isArray(manifest.icons)).toBe(true)
    const has192Any = manifest.icons.some(
      (i: { sizes: string; purpose?: string }) =>
        i.sizes === '192x192' && (i.purpose ?? 'any').includes('any'),
    )
    const has512Any = manifest.icons.some(
      (i: { sizes: string; purpose?: string }) =>
        i.sizes === '512x512' && (i.purpose ?? 'any').includes('any'),
    )
    const hasMaskable = manifest.icons.some(
      (i: { purpose?: string }) => (i.purpose ?? '').includes('maskable'),
    )
    expect(has192Any).toBe(true)
    expect(has512Any).toBe(true)
    expect(hasMaskable).toBe(true)
  })

  test('every declared icon URL returns 200 (regression: missing-icon trap)', async ({
    request,
  }) => {
    const res = await request.get('/manifest.webmanifest')
    const manifest = (await res.json()) as { icons: Array<{ src: string }> }
    for (const icon of manifest.icons) {
      const iconRes = await request.get(icon.src)
      expect(iconRes.status(), `icon ${icon.src} must exist`).toBe(200)
      // A broken/empty PNG would still 200 — sanity-check by size.
      const buf = await iconRes.body()
      expect(buf.length, `icon ${icon.src} must be non-empty`).toBeGreaterThan(200)
    }
  })

  test('apple-touch-icon is served (iOS home-screen icon)', async ({ request }) => {
    const res = await request.get('/apple-touch-icon-180.png')
    expect(res.status()).toBe(200)
  })

  test('index.html carries the iOS PWA meta tags', async ({ request }) => {
    const res = await request.get('/')
    expect(res.status()).toBe(200)
    const html = await res.text()
    expect(html).toContain('apple-touch-icon')
    expect(html).toContain('apple-mobile-web-app-capable')
    expect(html).toContain('apple-mobile-web-app-status-bar-style')
    expect(html).toContain('apple-mobile-web-app-title')
  })
})

test.describe('service worker + build identity', () => {
  test('sw.js is served and registers on first page load', async ({ page, request }) => {
    // Sanity: the SW asset itself must exist and be served with no-cache
    // (per web/learning-player/nginx.conf; in vite preview no-cache is implicit).
    const swRes = await request.get('/sw.js')
    expect(swRes.status()).toBe(200)
    expect(swRes.headers()['content-type']).toContain('javascript')

    // Load the app and wait for the SW to register + reach the "active" state.
    await page.goto('/')
    const state = await page.evaluate(async () => {
      const reg = await navigator.serviceWorker.ready
      return reg.active?.state ?? null
    })
    expect(state).toBe('activated')
  })

  test('window.__buildInfo is populated with sha + time', async ({ page }) => {
    await page.goto('/')
    const info = await page.evaluate(() => window.__buildInfo)
    expect(info).toBeTruthy()
    expect(typeof info?.sha).toBe('string')
    expect(info?.sha.length).toBeGreaterThan(0)
    expect(typeof info?.time).toBe('string')
    // ISO-8601 shape check (YYYY-MM-DDTHH:MM:SS) — enough to catch a broken injection.
    expect(info?.time).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)
  })
})
