import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * V2-C.1 / PRD-047 FR7.2: every show and episode labels its language wherever its metadata renders.
 *
 * The fixture corpus is genuinely multilingual — p01..p09 English, p10 Spanish, p11 Italian, p12
 * French, p13 German, p14 Portuguese — so FR7.5's "only when the corpus holds more than one language"
 * holds here and the badges must render. The single-language case (no badges at all) is the unit
 * test's job: this corpus cannot be made monolingual without forking the fixture.
 *
 * The badge's visible text is the lowercase code, uppercased by CSS; `data-lang` is asserted rather
 * than rendered text so a font or text-transform change cannot pass or fail this by accident.
 */

const badge = (scope: Locator | Page) => scope.getByTestId('language-badge')

/**
 * The V2-C.1 contrast check for badges ON THE PAGE: axe's colour-contrast rule, which can measure
 * them because their background is the page's. Requires axe to have PASSED at least one badge — an
 * `include` matching nothing, or a badge axe could not judge, would otherwise pass vacuously.
 */
async function expectBadgesReadable(page: Page): Promise<void> {
  const result = await new AxeBuilder({ page })
    .include('[data-testid="language-badge"]')
    .withRules(['color-contrast'])
    .analyze()
  expect(result.violations.flatMap((v) => v.nodes.map((n) => n.failureSummary))).toEqual([])
  expect(result.passes.flatMap((r) => r.nodes).length).toBeGreaterThan(0)
}

/**
 * The contrast check for badges OVER ARTWORK, which axe cannot do: the background is an image, so
 * it reports them "incomplete" and a white-on-white plate sailed through (measured by mutation,
 * 2026-10-08). Instead take the worst case the artwork could be — pure white, then pure black —
 * composite the plate's own translucent colour over it, and require AA (4.5:1) against the text.
 * The canvas does the colour parsing and compositing, so any CSS colour syntax works.
 */
async function expectOverlayBadgesReadable(scope: Locator): Promise<void> {
  const ratios = await scope
    .locator('[data-testid="language-badge"]')
    .evaluateAll((els) => {
      const ctx = document.createElement('canvas').getContext('2d', { willReadFrequently: true })!
      const paint = (under: string, over: string): number[] => {
        ctx.clearRect(0, 0, 1, 1)
        ctx.fillStyle = under
        ctx.fillRect(0, 0, 1, 1)
        ctx.fillStyle = over
        ctx.fillRect(0, 0, 1, 1)
        return [...ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)]
      }
      const lum = ([r, g, b]: number[]) =>
        [r, g, b]
          .map((c) => c / 255)
          .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
          .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0)
      const contrast = (a: number[], b: number[]) => {
        const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x)
        return (hi + 0.05) / (lo + 0.05)
      }
      return els.map((el) => {
        const cs = getComputedStyle(el)
        return Math.min(
          ...['#ffffff', '#000000'].map((art) => {
            const plate = paint(art, cs.backgroundColor)
            return contrast(paint(`rgb(${plate.join(',')})`, cs.color), plate)
          }),
        )
      })
    })
  expect(ratios.length).toBeGreaterThan(0)
  for (const ratio of ratios) expect(ratio).toBeGreaterThanOrEqual(4.5)
}

function visibleFeedMeta(page: Page): Locator {
  // The show page renders its meta line twice, one copy per breakpoint; match whichever is shown.
  return page.locator(
    '[data-testid="podcast-feed-meta"]:visible, [data-testid="podcast-feed-meta-wide"]:visible',
  )
}

test('the show page labels the show with its language', async ({ page }, testInfo) => {
  await signInIsolated(page, 'language-badge-show', testInfo)

  await page.goto('/podcast/p10')
  const spanish = badge(visibleFeedMeta(page))
  await expect(spanish).toHaveAttribute('data-lang', 'es')
  await expect(spanish).toHaveAttribute('aria-label', 'Language: Spanish')
  await expectBadgesReadable(page)

  // English is labelled too: in a multilingual corpus "en" is information, not a constant.
  await page.goto('/podcast/p05')
  await expect(badge(visibleFeedMeta(page))).toHaveAttribute('data-lang', 'en')
})

test('every show in Browse carries its language, as a tile and as a row', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'language-badge-browse', testInfo)
  await page.goto('/browse?tab=shows')

  const grid = page.getByTestId('show-browse-grid')
  const tiles = grid.locator('li')
  await expect(tiles.first()).toBeVisible()
  // Every tile, not just one: a show whose language never reached the client would be a hole here.
  await expect(badge(grid)).toHaveCount(await tiles.count())
  await expect(
    badge(tiles.filter({ hasText: 'Sesiones de Sendero' })),
  ).toHaveAttribute('data-lang', 'es')
  await expect(badge(tiles.filter({ hasText: 'Pfadgespräche' }))).toHaveAttribute('data-lang', 'de')
  await expectOverlayBadgesReadable(grid)

  await page.getByTestId('show-view').click()
  await page.getByTestId('show-view-opt-list').click()
  const rows = page.getByTestId('show-browse-list').getByTestId('show-row')
  await expect(rows.first()).toBeVisible()
  await expect(badge(page.getByTestId('show-browse-list'))).toHaveCount(await rows.count())
  await expect(
    badge(rows.filter({ hasText: "Sentieri d'Autore" })),
  ).toHaveAttribute('data-lang', 'it')
})

test('an episode carries its language on its card and in the player header', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'language-badge-episode', testInfo)
  await page.goto('/podcast/p12')

  const card = page
    .getByTestId('episode-card')
    .filter({ hasText: /Construire Des Sentiers Qui Durent/ })
    .first()
  await expect(badge(card)).toHaveAttribute('data-lang', 'fr')

  await card.getByRole('link', { name: /Construire Des Sentiers Qui Durent/ }).click()
  const heading = page.getByRole('heading', { name: /Construire Des Sentiers Qui Durent/ })
  await expect(heading).toBeVisible()
  // The facts line directly under the title: language, then date · duration.
  const facts = heading.locator('xpath=following-sibling::div[1]')
  await expect(badge(facts)).toHaveAttribute('data-lang', 'fr')
  await expect(badge(facts)).toHaveAttribute('aria-label', 'Language: French')
  await expectBadgesReadable(page)
})
