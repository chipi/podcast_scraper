import { expect, test, type Locator } from '@playwright/test'
import { signInIsolated } from './helpers'

/** A fresh account per RUN: these tests follow, save and delete, and an account reused from the
 *  last run starts in the state that run left (already Following) — a flip test then flips it off. */
const RUN = Date.now().toString(36)

/**
 * Shows and episodes on ONE page act alike (operator 2026-10-05, UXS-014 §ShowRow, UXS-012 §What's
 * new). Geometry is asserted from real boxes, so a layout that renders the right controls in the
 * wrong place fails here — which a unit test under happy-dom (no layout) cannot see.
 */

const box = async (l: Locator) => {
  const b = await l.boundingBox()
  expect(b, 'element has no box — not rendered').not.toBeNull()
  return b!
}

test.beforeEach(async ({ page }, testInfo) => {
  await signInIsolated(page, `controls-${testInfo.title.slice(0, 20)}-${RUN}`, testInfo)
})

test('Library › Saved: a show row puts its controls UNDER the artwork, like the episode rows', async ({
  page,
}) => {
  const r = await page.request.put('/api/app/favorites', {
    data: { kind: 'show', ref: 'p05', label: 'Long Horizon Notes' },
  })
  expect(r.ok()).toBe(true)
  await page.goto('/library?tab=saved')
  const row = page.getByTestId('saved-shows-list').getByTestId('show-row').first()
  await expect(row).toBeVisible()
  const art = await box(row.locator('img').first())
  const controls = row.getByTestId('show-row-actions')
  await expect(controls.getByTestId('favorite-button')).toBeVisible()
  const c = await box(controls)
  expect(c.y, 'controls must start below the artwork').toBeGreaterThanOrEqual(art.y + art.height - 1)
  expect(c.x, 'controls sit in the artwork column').toBeLessThan(art.x + art.width)
})

test('Discover › Shows (list): Follow + heart under the artwork; the pill reads exactly Follow / Following', async ({
  page,
}) => {
  await page.goto('/browse?tab=shows')
  await page.getByTestId('show-view').click()
  await page.getByTestId('show-view-opt-list').click()
  const row = page.getByTestId('show-browse-list').getByTestId('show-row').first()
  const art = await box(row.locator('img').first())
  const controls = row.getByTestId('show-row-actions')
  const follow = controls.getByTestId('follow-show')
  await expect(follow).toHaveText('Follow') // no "+", no "show"
  const c = await box(controls)
  expect(c.y).toBeGreaterThanOrEqual(art.y + art.height - 1)
  // Follow and the heart fit side by side in the 128px column — one line.
  const f = await box(follow)
  const heart = await box(controls.getByTestId('favorite-button'))
  expect(Math.abs(f.y + f.height / 2 - (heart.y + heart.height / 2))).toBeLessThan(6)
  await follow.click()
  await expect(follow).toHaveText('Following')
  await follow.click()
  await expect(follow).toHaveText('Follow')
})

test('Search: a show row carries ONE ⋯ right of its name, holding Follow / Save / Add to board', async ({
  page,
}) => {
  await page.goto('/search?q=Long%20Horizon')
  const row = page.getByTestId('search-shows').getByTestId('show-row').first()
  await expect(row).toBeVisible({ timeout: 30_000 })
  // Nothing on or under the artwork here — the ⋯ is the row's only control.
  await expect(row.getByTestId('show-row-actions')).toHaveCount(0)
  const name = await box(row.locator('a[href^="/podcast/"]').first())
  const menu = row.getByTestId('show-row-menu')
  const m = await box(menu)
  expect(m.x, 'the ⋯ sits right of the name').toBeGreaterThanOrEqual(name.x + name.width - 1)
  expect(m.y, 'on the name\'s line').toBeLessThan(name.y + name.height)

  await menu.getByTestId('overflow-trigger').click()
  const panel = page.getByTestId('overflow-menu')
  await expect(panel.getByRole('menuitem')).toHaveCount(3)
  await expect(panel.getByTestId('follow-show')).toContainText('Follow')
  await expect(panel.getByTestId('favorite-button')).toContainText('Save')
  await expect(panel.getByRole('menuitem', { name: /Add to board/ })).toBeVisible()

  // Following from the menu is real: the show page agrees after a reload.
  await panel.getByTestId('follow-show').click()
  await page.goto('/podcast/p05')
  await expect(page.getByTestId('follow-show').first()).toHaveText('Following')
})

test("What's new: every position carries ♡ queue ⋯ — a row on #01, one column on 02+", async ({
  page,
}) => {
  await page.goto('/')
  const section = page.getByRole('heading', { name: "What's new" }).locator('xpath=ancestor::section[1]')
  const rows = section.getByTestId('episode-actions')
  await expect(rows.nth(1)).toBeVisible()
  // Home's other sections load asynchronously and push What's new down as they land, so boxes
  // measured one call apart can straddle a reflow (it read a 98px "gap" between two siblings).
  // Wait for the page to settle, then read every box in ONE frame.
  await page.waitForLoadState('networkidle')
  const layout = await rows.evaluateAll((els) =>
    els.map((el) => {
      const r = (sel: string) => {
        const b = (el.querySelector(sel) as HTMLElement).getBoundingClientRect()
        return { x: b.x, y: b.y }
      }
      const queue = Array.from(el.querySelectorAll('button')).find((b) =>
        /queue/i.test(b.getAttribute('aria-label') ?? ''),
      ) as HTMLElement
      const q = queue.getBoundingClientRect()
      const li = el.closest('li')
      const title = li?.querySelector('a[href^="/episode/"]')?.getBoundingClientRect()
      return {
        heart: r('[data-testid="favorite-button"]'),
        queue: { x: q.x, y: q.y },
        more: r('[data-testid="overflow-trigger"]'),
        titleShare: li && title ? title.width / li.getBoundingClientRect().width : null,
      }
    }),
  )
  expect(layout.length).toBeGreaterThanOrEqual(2)
  layout.forEach(({ heart, queue, more }, i) => {
    if (i === 0) {
      // #01: one row, top right of the card.
      expect(Math.abs(heart.y - queue.y), '#01 heart/queue not on one row').toBeLessThan(4)
      expect(Math.abs(queue.y - more.y), '#01 queue/⋯ not on one row').toBeLessThan(4)
      expect(heart.x).toBeLessThan(queue.x)
      expect(queue.x).toBeLessThan(more.x)
    } else {
      // 02+: one column, top to bottom.
      expect(Math.abs(heart.x - queue.x), `row ${i + 1} not one column`).toBeLessThan(4)
      expect(Math.abs(queue.x - more.x), `row ${i + 1} not one column`).toBeLessThan(4)
      expect(heart.y).toBeLessThan(queue.y)
      expect(queue.y).toBeLessThan(more.y)
    }
  })
  // The stack costs height, not width: a ranked row's title keeps most of the row.
  expect(layout[1].titleShare).toBeGreaterThan(0.5)
})

test('an episode card shows the publisher\'s description, not our summary', async ({ page }) => {
  await page.goto('/podcast/p05')
  const card = page.getByTestId('episode-card').filter({ hasText: 'Index Investing Without the Myths' }).first()
  await expect(card).toBeVisible()
  await expect(card).toContainText('What indexing does well')
  await expect(card).not.toContainText('Index funds are not a strategy')
})
