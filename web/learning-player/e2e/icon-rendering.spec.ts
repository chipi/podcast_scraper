import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Icons must actually RENDER — and be drawn, not typed (UXS-014).
 *
 * Four bugs this session were one bug: a codepoint used as an icon became a tofu box on iOS,
 * because the glyph is missing from the platform UI font. Every one was found by a human looking at
 * a screenshot, because the elements are `aria-hidden` and so invisible to XCUITest, which reads
 * the accessibility tree.
 *
 * `aria-hidden` is CORRECT here and is not being changed: the interactive ancestor carries the
 * accessible name, so announcing the graphic as well would duplicate it. The mistake was testing
 * through the a11y tree. Playwright queries the DOM, so it sees these elements fine — verified:
 * a hidden `<svg>` reports visible with a real 16x16 box.
 *
 * Two assertions per control, and the pair is the point:
 *   - an `<svg>` exists and has NON-ZERO size — a missing or collapsed icon fails
 *   - the control renders no tofu-prone CHARACTER — a regression to a codepoint fails
 * The unit guard (`src/__checks__/icon-glyphs.test.ts`) catches the codepoint in source; this
 * catches it in a real browser, plus the case that guard cannot see: an icon that is present in the
 * markup but renders as nothing.
 */
const TOFU = ['✕', '✎', '►', '＋', '🔥']

async function assertDrawnIcon(scope: import('@playwright/test').Locator, what: string) {
  const svg = scope.locator('svg').first()
  await expect(svg, `${what}: no <svg> — is it still a text glyph?`).toBeVisible()
  const box = await svg.boundingBox()
  expect(box, `${what}: icon has no box`).not.toBeNull()
  expect(box!.width, `${what}: icon collapsed to zero width`).toBeGreaterThan(4)
  expect(box!.height, `${what}: icon collapsed to zero height`).toBeGreaterThan(4)

  const text = (await scope.textContent()) ?? ''
  for (const glyph of TOFU) {
    expect(text, `${what}: renders the character ${glyph}, which is tofu on iOS`).not.toContain(
      glyph
    )
  }
}

test('icon-only controls render a drawn icon, not a character', async ({ page }, testInfo) => {
  await signInIsolated(page, 'icon-rendering', testInfo)
  await page.goto('/')

  // The episode action row's overflow — an icon-only button with its name on the button.
  const overflow = page.getByTestId('overflow-trigger').first()
  await expect(overflow).toBeVisible()
  await assertDrawnIcon(overflow, 'overflow trigger')
})

test('a sheet dismiss control renders a drawn ✕', async ({ page }, testInfo) => {
  await signInIsolated(page, 'icon-dismiss', testInfo)
  await page.goto('/podcast/p05')

  // Open an entity card from the show's signals band; its dismiss is the control that was showing
  // "?" on device for every sheet in the app.
  const chip = page.getByTestId('ps-distinctive-topic').first()
  await expect(chip).toBeVisible()
  await chip.click()

  const dismiss = page.getByTestId('ec-dismiss').first()
  await expect(dismiss).toBeVisible()
  await assertDrawnIcon(dismiss, 'entity card dismiss')
})

test('every share control wears the same share icon (operator 2026-10-07)', async ({ page }, testInfo) => {
  // Highlights drew a bare "↗", which reads as "open externally". One glyph for every share.
  await signInIsolated(page, 'icon-share', testInfo)
  const eps = await page.request.get('/api/app/episodes?page_size=1')
  const slug = (await eps.json()).items?.[0]?.slug
  expect(slug, 'fixture corpus returned no episode').toBeTruthy()
  const made = await page.request.post('/api/app/highlights', {
    data: { episode_slug: slug, kind: 'moment', start_ms: 1000 },
  })
  expect(made.ok(), `seeding a highlight failed: ${made.status()}`).toBeTruthy()

  await page.goto(`/episode/${encodeURIComponent(slug)}`)
  const menu = page.getByTestId('share-menu').first()
  await expect(menu).toBeVisible()
  await expect(menu.getByTestId('share-icon')).toHaveCount(1)
  await assertDrawnIcon(menu, 'episode share')

  await page.goto('/library?tab=saved')
  const hl = page.getByTestId('highlight-share').first()
  await expect(hl).toBeVisible()
  await expect(hl.getByTestId('share-icon')).toHaveCount(1)
  await assertDrawnIcon(hl, 'highlight share')
  expect((await hl.textContent()) ?? '').not.toContain('↗')
})
