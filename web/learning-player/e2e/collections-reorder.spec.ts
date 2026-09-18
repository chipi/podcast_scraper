import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Drag-reorder on the Boards list — the control had NO coverage of any kind (operator 2026-09-18).
 *
 * The operator reported "dragging collections to set order never started working". Nothing in the
 * suite touched `collection-drag-handle`, so there was no way to tell whether it had ever worked,
 * and the first attempt at a fix was a guess dressed up as a diagnosis: a screenshot showed iOS
 * text-selection handles, so `user-select: none` was added on the theory that `touch-action` does
 * not suppress iOS's selection gesture. That may or may not be true. It was never tested.
 *
 * This drives the real pointer sequence — down, moves, up — against a real API and asserts the order
 * actually changed AND survives a reload. Chromium is not iOS, so this cannot prove the device
 * behaviour. What it CAN do is tell us whether the drag works at all: if it fails here, the cause is
 * not iOS-specific and no amount of selection suppression was ever going to fix it.
 *
 * Moves in small steps rather than one jump. `onGrabMove` picks the nearest row mid-point, and a
 * single large move can be delivered as one event that the browser coalesces — which is the shape of
 * a test that passes while a human's slower drag does not.
 */
test('a board can be dragged into a new position, and the order persists', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'collections-reorder', testInfo)

  const run = `${Date.now().toString(36)}`
  const names = [`Alpha ${run}`, `Bravo ${run}`, `Charlie ${run}`]

  // Three boards, created newest-last so the default `updated` sort gives a known starting order.
  await page.goto('/browse')
  await page.waitForLoadState('networkidle')
  const firstRow = page.locator('article').first()
  await expect(firstRow).toBeVisible()

  for (const name of names) {
    await firstRow.getByTestId('overflow-trigger').click()
    await page.getByTestId('add-to-collection').click()
    const menu = page.getByTestId('add-to-collection-menu')
    await expect(menu).toBeVisible()
    await menu.locator('input').fill(name)
    await menu.locator('form button[type="submit"]').click()
    await expect(page.getByTestId('add-to-collection-menu')).toBeHidden({ timeout: 5000 })
    await page.keyboard.press('Escape')
  }

  await page.goto('/library?tab=collections')
  await expect(page.getByTestId('collections-load-error')).toHaveCount(0)

  const rows = page.locator('[data-board-row]')
  await expect(rows.first()).toBeVisible()

  // The handle must EXIST before anything else is worth asserting. It renders only while the list is
  // in its manual order (`sortBy === 'updated'` and no active search), which is the default — if it
  // is absent here, the drag is unreachable for reasons that have nothing to do with the gesture.
  const handles = page.getByTestId('collection-drag-handle')
  await expect(
    handles.first(),
    'no drag handle on the boards list — the control is unreachable, so the gesture never runs',
  ).toBeVisible()

  const orderOf = async () =>
    (await rows.evaluateAll((els) => els.map((el) => (el as HTMLElement).dataset.boardRow ?? ''))) //
      .filter(Boolean)

  const before = await orderOf()
  expect(before.length, 'need at least two boards to reorder').toBeGreaterThan(1)

  // Drag the FIRST board down past the second.
  const source = handles.first()
  const sourceBox = (await source.boundingBox())!
  const targetBox = (await rows.nth(1).boundingBox())!

  await page.mouse.move(sourceBox.x + sourceBox.width / 2, sourceBox.y + sourceBox.height / 2)
  await page.mouse.down()
  const startY = sourceBox.y + sourceBox.height / 2
  const endY = targetBox.y + targetBox.height * 0.75
  for (let step = 1; step <= 8; step++) {
    await page.mouse.move(
      sourceBox.x + sourceBox.width / 2,
      startY + ((endY - startY) * step) / 8,
    )
  }
  await page.mouse.up()

  const after = await orderOf()
  expect(
    after,
    'the drag did not change the order — pointerdown/move/up ran but the list is unmoved',
  ).not.toEqual(before)
  expect(after[0], 'the dragged board did not leave the first position').not.toBe(before[0])

  // And it is PERSISTED, not just reordered in local state. A reorder that only lives in the view
  // reads to the user as the drag having failed the moment they come back.
  await page.reload()
  await expect(rows.first()).toBeVisible()
  expect(
    await orderOf(),
    'the new order did not survive a reload — the reorder was never written to the server',
  ).toEqual(after)
})
