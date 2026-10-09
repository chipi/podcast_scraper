import { expect, test, type Page } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * Where a tapped new-episodes push lands (2026-10-09).
 *
 * The push carries `url`: the episode when it announces one, Home's What's new (`/#whats-new`) when
 * it announces several. The web service worker opens that URL in a tab (a full load); the native
 * app routes to it in-app (services/pushTaps.ts). Both must land with What's new on screen — on a
 * phone it sits below the hero, so landing at the top of Home would leave the reader looking for it.
 */

/**
 * On screen once Home has SETTLED: the sections above it (the welcome, search) render after the
 * first paint and push it down, so a check on the first frame can pass on a page that then moves.
 */
async function expectWhatsNewOnScreen(page: Page): Promise<void> {
  const heading = page.getByTestId('home-whats-new').getByRole('heading', { name: "What's new" })
  await expect(heading).toBeVisible()
  await page.waitForLoadState('networkidle')
  await expect(heading).toBeInViewport()
}

test.describe("a push about several episodes opens What's new", () => {
  test.use({ viewport: { width: 390, height: 600 } })

  test('from a cold open of the link (web push)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'push-open-cold', testInfo)
    // A NEW document, as the service worker's openWindow / navigate gives. Sign-in leaves the page
    // on `/`, so going straight to `/#whats-new` would be a same-document hash change instead.
    await page.goto('about:blank')
    await page.goto('/#whats-new')
    await expectWhatsNewOnScreen(page)
  })

  test('from inside the running app (native tap)', async ({ page }, testInfo) => {
    await signInIsolated(page, 'push-open-warm', testInfo)
    await navTo(page, 'library')
    // Exactly what the tap handler does: `router.push(url)`, no reload. Vue keeps the app on its
    // mount element in production builds too.
    await page.evaluate(async () => {
      const el = document.querySelector('#app') as unknown as {
        __vue_app__: { config: { globalProperties: { $router: { push: (p: string) => Promise<unknown> } } } }
      }
      await el.__vue_app__.config.globalProperties.$router.push('/#whats-new')
    })
    await expect(page).toHaveURL(/\/#whats-new$/)
    await expectWhatsNewOnScreen(page)
  })
})
