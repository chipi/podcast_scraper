import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Home's two PERSONAL surfaces, against a real api: the boards teaser and the revisit rail.
 *
 * ## Why they need their own spec
 *
 * Both were shipped with unit tests and no e2e, and both broke in ways no unit test could see. The
 * revisit rail rendered NOTHING for its first hour of existence because its import silently failed
 * to land — Vue treats an unresolved component as a runtime warning, so `npm run build` printed a
 * tick and 1512 unit tests passed over a blank surface. The boards teaser then deep-linked to
 * `?board=<id>`, which nothing consumed, and wiring that up put a `watch` above the `const`s it
 * read: a temporal dead zone that took down the ENTIRE Collections view, again with a green build
 * and a green suite.
 *
 * The common thread is that neither the build nor the unit suite can see runtime wiring. Only
 * loading the page can.
 */

test('Home shows your boards, and a tile opens that board expanded', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-boards-teaser', testInfo)

  // Unique per run: `signInIsolated` seeds a stable account, so boards accumulate across runs and
  // any assertion on totals would fail on the third run for a reason that looks like a bug.
  const name = `Home teaser ${Date.now().toString(36)}`

  // --- 1. create a board with a member, so it has a cover and a count -------------------------
  await page.goto('/browse')
  await page.waitForLoadState('networkidle')
  const firstRow = page.locator('article').first()
  await expect(firstRow).toBeVisible()

  await firstRow.getByTestId('overflow-trigger').click()
  await page.getByTestId('add-to-collection').click()
  const menu = page.getByTestId('add-to-collection-menu')
  await expect(menu).toBeVisible()
  await menu.locator('input').fill(name)
  await menu.locator('form button[type="submit"]').click()
  await expect(menu).toBeHidden({ timeout: 5000 })
  await page.keyboard.press('Escape')

  // --- 2. it appears on Home -------------------------------------------------------------------
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  const teaser = page.getByTestId('home-collections-teaser')
  await expect(
    teaser,
    'the boards teaser did not render on Home — the surface a build cannot prove exists',
  ).toBeVisible()
  // Most recently CHANGED first, and this board was just touched, so it leads.
  await expect(teaser.getByTestId('home-collection-tile').first()).toContainText(name)

  // --- 3. the tile opens THAT board, expanded ---------------------------------------------------
  // The Boards list is an accordion. Landing on it collapsed would make the tile feel inert, which
  // is exactly what it did before the deep link was consumed.
  await teaser.getByTestId('home-collection-tile').first().click()
  await page.waitForURL(/tab=collections/)
  await expect(page).toHaveURL(/board=/)

  const openRow = page.locator('[data-board-row]').filter({ hasText: name }).first()
  await expect(openRow).toBeVisible()
  // An open board renders its members inside its own row, so it is far taller than a collapsed
  // one. Asserting on the CONTENT rather than a pixel height: the row carries the episode it holds.
  await expect(
    openRow.locator('a[href^="/episode/"]').first(),
    'the deep-linked board landed collapsed',
  ).toBeVisible()
})

test('the revisit rail stays absent while nothing is due', async ({ page }, testInfo) => {
  await signInIsolated(page, 'home-revisit-empty', testInfo)

  /**
   * The honest assertion available here, and it is not a throwaway.
   *
   * A capture is not due until the ladder's first rung (2 days), and an e2e run creates captures
   * seconds ago — so a POPULATED rail cannot be produced without either waiting two days or
   * writing backdated state behind the app's back, which would test the fixture rather than the
   * product. What IS testable, and worth pinning, is that an empty queue renders NOTHING: the
   * section is `v-if`'d on having items, and a rail that appeared empty on a new user's Home would
   * be the app asking for work they have not created yet.
   *
   * The populated behaviour — one card per episode, drop-and-backfill on both actions, restore on
   * failure — is covered in `src/components/RevisitRail.test.ts`, where the store can be driven
   * directly.
   */
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  await expect(page.getByTestId('home-revisit-rail')).toHaveCount(0)

  // And the surface it points at agrees, rather than the two disagreeing about the same question.
  await page.goto('/library?tab=revisit')
  await expect(page.getByText(/Nothing to revisit right now/)).toBeVisible()
})


test('every Home section header is the same shape', async ({ page }, testInfo) => {
  /**
   * Home had grown three headers by hand — kicker above the title on some sections and beside it on
   * others, three title fonts (`h1` display, `h2` display, `lp-section`), and nothing stopping
   * either line wrapping to a second row (operator 2026-09-18). `SectionHeading` now owns the
   * shape; this asserts the shape rather than the component, so hand-rolling a seventh variant
   * fails here even if it never imports the component.
   */
  await signInIsolated(page, 'home-heading-uniformity', testInfo)
  await page.goto('/')
  await page.waitForLoadState('networkidle')

  const titles = page.getByTestId('section-title')
  await expect(titles.first()).toBeVisible()

  const shape = await titles.evaluateAll((els) =>
    els.map((el) => {
      const cs = getComputedStyle(el)
      const r = el.getBoundingClientRect()
      return {
        text: (el.textContent ?? '').trim(),
        left: Math.round(r.left),
        fontSize: cs.fontSize,
        // A wrapped heading is taller than one line of its own line-height.
        wrapped: r.height > parseFloat(cs.lineHeight) + 2,
      }
    }),
  )

  expect(shape.length, 'no shared section headings rendered').toBeGreaterThan(2)
  expect(
    [...new Set(shape.map((s) => s.fontSize))],
    'section titles render at more than one size',
  ).toHaveLength(1)
  expect(
    [...new Set(shape.map((s) => s.left))],
    'section titles do not share a left edge',
  ).toHaveLength(1)
  expect(
    shape.filter((s) => s.wrapped).map((s) => s.text),
    'a section title wrapped to a second row',
  ).toEqual([])

  // The kicker is a count or a date — never the title again in other words. Two did exactly that
  // ("For you" over "Your Week"; "Ask across every episode" over "Find any moment you've heard.").
  const kickers = await page.getByTestId('section-kicker').allTextContents()
  for (const k of kickers) {
    expect(k, `kicker "${k}" carries no number or date`).toMatch(/\d/)
  }
})
