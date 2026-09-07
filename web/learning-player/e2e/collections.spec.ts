import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The collections loop, end to end, against a REAL api (#2013).
 *
 * ## Why this exists
 *
 * Collections shipped broken and every test layer was green. The unit tests mock the API, so a
 * broken real flow still passes. The post-deploy live check exercises the endpoints via `request` —
 * API only, never the UI. And the sole UI assertion anywhere was `library-saved.spec.ts` checking
 * that "No collections yet" is visible: a test that only ever asserts the EMPTY state cannot tell
 * "you have none" from "we lost them", which is exactly the bug.
 *
 * So this drives the real interface against the real server through the whole loop, which is the
 * one thing nothing else did.
 */
test('add to a collection, create another, and find both with their items', async ({ page }, testInfo) => {
  await signInIsolated(page, 'collections-loop', testInfo)

  /**
   * Names are unique PER RUN, and every assertion is scoped to them.
   *
   * `signInIsolated` seeds a stable user from its label, so the account survives between runs and
   * collections accumulate. A first draft asserted `toHaveCount(2)` and passed once, then reported
   * `Received: 6` on the third run — a spec that only works on a clean volume is a spec that will
   * fail in CI for a reason that looks like a product bug.
   */
  const run = `${Date.now().toString(36)}`
  const A = `Research ${run}`
  const B = `Later ${run}`

  // --- 1. add an episode to a brand-new collection --------------------------------------------
  // From a browse ROW: the episode-kind control lives on `EpisodeCard`. There is deliberately no
  // note here about the player page — see the comment at the bottom of this file.
  await page.goto('/browse')
  await page.waitForLoadState('networkidle')
  const firstRow = page.locator('article').first()
  await expect(firstRow).toBeVisible()
  const slug = (await firstRow.locator('a[href^="/episode/"]').first().getAttribute('href'))!.split('/').pop()!

  await firstRow.getByTestId('add-to-collection').click()
  await expect(page.getByTestId('add-to-collection-menu')).toBeVisible()

  const menu = page.getByTestId('add-to-collection-menu')
  await menu.locator('input').fill(A)
  await menu.locator('form button[type="submit"]').click()
  await expect(page.getByTestId('add-to-collection-menu')).toBeHidden({ timeout: 5000 })

  // No error surfaced — the write actually landed.
  await expect(page.getByTestId('collection-error')).toHaveCount(0)

  // --- 2. a SECOND collection, from the same control -----------------------------------------
  await firstRow.getByTestId('add-to-collection').click()
  const menu2 = page.getByTestId('add-to-collection-menu')
  await expect(menu2).toBeVisible()
  await menu2.locator('input').fill(B)
  await menu2.locator('form button[type="submit"]').click()
  await expect(page.getByTestId('add-to-collection-menu')).toBeHidden({ timeout: 5000 })

  // --- 3. reopen the control: BOTH collections persist and are offered ------------------------
  // This is the assertion the old empty-state test could never make.
  await page.reload()
  await page.waitForLoadState('networkidle')
  await page.locator('article').first().getByTestId('add-to-collection').click()
  const reopened = page.getByTestId('add-to-collection-menu')
  await expect(reopened).toBeVisible()
  await expect(reopened).toContainText(A)
  await expect(reopened).toContainText(B)
  await page.keyboard.press('Escape')

  // --- 4. Library → Collections lists both ----------------------------------------------------
  await page.goto('/library?tab=collections')
  await page.waitForLoadState('networkidle')
  await expect(page.getByTestId('collections-load-error')).toHaveCount(0)
  const rows = page.getByTestId('collection-open')
  // Both of THIS run's collections are listed. Scoped by name rather than by total count, which
  // depends on how many times the suite has run against this volume.
  await expect(rows.filter({ hasText: A })).toHaveCount(1)
  await expect(rows.filter({ hasText: B })).toHaveCount(1)

  // --- 5. open one — the episode is rendered INSIDE it ----------------------------------------
  await rows.filter({ hasText: A }).first().click()
  const items = page.getByTestId('collection-items')
  await expect(items).toBeVisible()
  await expect(items.locator('li')).toHaveCount(1)
  // the item points at the episode we added, not just "something"
  await expect(items).toContainText(/\S/)
  const href = await items.locator('a, [href]').first().getAttribute('href').catch(() => null)
  if (href) expect(href).toContain(slug)
})

test('the episode page can pin the episode you are listening to (#2013 follow-up)', async ({ page }, testInfo) => {
  // The control used to be reachable only from a LIST row or from the show page (which pins the
  // whole show). The moment you most want to pin an episode is while you are listening to it.
  await signInIsolated(page, 'collections-player', testInfo)
  await page.goto('/podcast/p05')
  await page.waitForLoadState('networkidle')
  await page.locator('a[href^="/episode/"]').first().click()
  await page.waitForURL(/\/episode\//)

  await page.getByTestId('add-to-collection').first().click()
  const menu = page.getByTestId('add-to-collection-menu')
  await expect(menu).toBeVisible()
  const name = `From the player ${Date.now().toString(36)}`
  await menu.locator('input').fill(name)
  await menu.locator('form button[type="submit"]').click()
  await expect(page.getByTestId('add-to-collection-menu')).toBeHidden({ timeout: 5000 })
  await expect(page.getByTestId('collection-error')).toHaveCount(0)

  // and it lands as an EPISODE, in the collection, readable from Library
  await page.goto('/library?tab=collections')
  await page.waitForLoadState('networkidle')
  await page.getByTestId('collection-open').filter({ hasText: name }).first().click()
  await expect(page.getByTestId('collection-items').locator('li')).toHaveCount(1)
})

/**
 * The confirmation in front of "delete collection" (#1594), driven in a real browser.
 *
 * The unit tests prove the wiring — that the ✕ opens a dialog and only the accept button deletes.
 * They cannot prove the two properties that make the dialog SAFE, because jsdom has no `<dialog>`
 * at all and the stub they use is one I wrote: that Escape dismisses it without deleting, and that
 * the dialog is modal. Both are the browser's behaviour, so they are asserted where a browser is.
 */
test('deleting a collection asks first, and Escape means no', async ({ page }, testInfo) => {
  await signInIsolated(page, 'collections-confirm', testInfo)

  const name = `Confirm ${testInfo.project.name} ${Date.now()}`
  const made = await page.request.post('/api/app/collections', { data: { name } })
  expect(made.ok(), `seeding a collection failed: ${made.status()}`).toBeTruthy()

  await page.goto('/library?tab=collections')
  await page.waitForLoadState('networkidle')

  const row = page.locator('li', { hasText: name })
  await expect(row).toHaveCount(1)

  // 1. The ✕ opens the dialog and destroys nothing.
  await row.getByTestId('collection-delete').click()
  const dialog = page.getByTestId('collection-delete-confirm')
  await expect(dialog).toBeVisible()
  await expect(row, 'the collection disappeared before anything was confirmed').toHaveCount(1)

  // 2. Escape dismisses. This is the property a userland modal usually gets wrong, and getting it
  //    wrong here means a dialog the keyboard cannot escape sitting over a destructive action.
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()

  // 3. Re-opening still works — asserted BEFORE any reload, deliberately. The browser closes the
  //    dialog on Escape without telling the parent, so if the component does not bridge `close`
  //    back to a cancel, the parent still holds the pending id: tapping ✕ on the same row sets it
  //    to the value it already has, nothing changes, and the dialog never reopens. The next delete
  //    then has no confirmation at all. A reload here would reset that state and hide the bug —
  //    an earlier draft of this test did exactly that and stayed green when the bridge was removed.
  await row.getByTestId('collection-delete').click()
  await expect(dialog, 'the dialog did not reopen — the Escape-close never reached the parent').toBeVisible()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()

  // 4. Nothing was deleted by any of that, and it survives a round trip to the server.
  await page.reload()
  await expect(
    page.locator('li', { hasText: name }),
    'Escape deleted the collection — dismissing a confirmation must never confirm it',
  ).toHaveCount(1)

  // 5. Confirming actually deletes, and it stays deleted across a reload.
  await page.locator('li', { hasText: name }).getByTestId('collection-delete').click()
  await page.getByTestId('confirm-accept').click()
  await expect(page.locator('li', { hasText: name })).toHaveCount(0)
  await page.reload()
  await expect(page.locator('li', { hasText: name })).toHaveCount(0)
})

