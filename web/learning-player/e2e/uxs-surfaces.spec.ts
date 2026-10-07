import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/** A fresh account per RUN: these tests follow, save and delete, and an account reused from the
 *  last run starts in the state that run left (already Following) — a flip test then flips it off. */
const RUN = Date.now().toString(36)

/**
 * UXS behaviours that had unit tests but no browser coverage (audit 2026-10-05): each one here was
 * only ever asserted under happy-dom, against mocked stores. One test per behaviour, named for the
 * spec line it proves.
 */
const EPISODE = 'Index Investing Without the Myths'

test.beforeEach(async ({ page }, testInfo) => {
  await signInIsolated(page, `uxs-${testInfo.title.slice(0, 22)}-${RUN}`, testInfo)
})

test('UXS-013 §3.4: a search that names a topic shows its card above the passages, and it opens', async ({
  page,
}) => {
  await page.goto('/search?q=risk%20management')
  await expect(page.getByTestId('search-section-topics')).toBeVisible({ timeout: 30_000 })
  const card = page.getByRole('button', { name: /^Open risk management/i })
  await expect(card).toBeVisible()
  // ABOVE the passages: the entity is what the reader named; passages are evidence for it.
  const episodes = page.getByTestId('search-section-episodes')
  if (await episodes.count()) {
    const a = await card.boundingBox()
    const b = await episodes.boundingBox()
    expect(a!.y).toBeLessThan(b!.y)
  }
  await card.click()
  const dialog = page.getByRole('dialog').filter({ hasText: /risk management/i }).first()
  await expect(dialog).toBeVisible()
  await expect(dialog.getByTestId('ec-dismiss')).toBeVisible()
})

test('UXS-012 §Theme: the theme page names its members, lists episodes newest first, and follows', async ({
  page,
}) => {
  await page.goto('/theme/tc:broadcast-format')
  const view = page.getByTestId('theme-view')
  await expect(view).toBeVisible()
  await expect(view).toContainText('broadcast format')
  // The member topics, each with its episode count.
  await expect(view.getByTestId('member-episodes').first()).toBeVisible()
  await expect(view.getByTestId('episodes-order')).toHaveText(/newest first/i)
  const follow = view.getByTestId('theme-follow')
  await expect(follow).toContainText('Follow')
  await follow.click()
  await expect(follow).toContainText('Following')
  // Persisted, not just flipped: the server has it after a reload.
  await page.reload()
  await expect(page.getByTestId('theme-view').getByTestId('theme-follow')).toContainText('Following')
})

test('UXS-013: following a topic from its page flips Follow → Following and persists', async ({ page }) => {
  await page.goto('/topic/topic:risk-management')
  const follow = page.getByTestId('ec-follow').first()
  await expect(follow).toContainText(/^\s*\+?\s*Follow\b/)
  await follow.click()
  await expect(follow).toContainText('Following')
  await page.reload()
  await expect(page.getByTestId('ec-follow').first()).toContainText('Following')
})

test('UXS-014 §Destructive: deleting a highlight asks first; Cancel keeps it, confirming removes it', async ({
  page,
}) => {
  const eps = await (await page.request.get('/api/app/podcasts/p05/episodes')).json()
  const slug = (eps as { items: { slug: string; title: string }[] }).items.find((e) =>
    e.title.startsWith('Index Investing'),
  )!.slug
  const quote = `delete me ${Date.now()}`
  await page.request.post('/api/app/highlights', {
    data: { episode_slug: slug, kind: 'span', start_ms: 5000, quote_text: quote },
  })
  await page.goto('/library?tab=saved')
  const card = page.getByTestId('highlight-card').filter({ hasText: quote })
  await expect(card).toBeVisible()

  await card.getByTestId('highlight-delete').click()
  const confirm = page.getByTestId('highlight-delete-confirm')
  await expect(confirm).toBeVisible()
  // The SAFE choice holds focus first, so a stray Enter cannot delete.
  // Saved keeps several confirm dialogs mounted (highlight, note, board) — scope to the open one.
  await expect(confirm.getByTestId('confirm-cancel')).toBeFocused()
  await confirm.getByTestId('confirm-cancel').click()
  await expect(card).toBeVisible()

  await card.getByTestId('highlight-delete').click()
  await page.getByTestId('highlight-delete-confirm').getByTestId('confirm-accept').click()
  await expect(page.getByTestId('highlight-card').filter({ hasText: quote })).toHaveCount(0)
})

test('UXS-014 §CollapsibleSection: a section you close in Episode notes stays closed after a reload', async ({
  page,
}) => {
  await page.goto('/podcast/p05')
  await page.getByText(EPISODE).first().click()
  await expect(page).toHaveURL(/\/episode\//)
  await page.getByTestId('player-open-insights').click()
  // A native <details>: its OPEN state is the truth (the summary carries no aria-expanded).
  const panel = page.getByTestId('knowledge-panel')
  const section = panel.getByTestId('kp-section-key-points')
  await expect(section).toHaveJSProperty('open', true)
  await panel.getByTestId('kp-section-toggle-key-points').click()
  await expect(section).toHaveJSProperty('open', false)
  // <details> fires `toggle` a task AFTER the click, and the preference is written from it — wait
  // for the write, or the reload races it (no reader reloads within milliseconds).
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem('lp.kp.key-points')))
    .toBe('closed')
  await page.reload()
  await page.getByTestId('player-open-insights').click()
  await expect(page.getByTestId('knowledge-panel').getByTestId('kp-section-key-points')).toHaveJSProperty(
    'open',
    false,
  )
  // Leave it open again for anyone reading this account's state.
  await page.getByTestId('knowledge-panel').getByTestId('kp-section-toggle-key-points').click()
})

test('UXS-012: "Browse all →" under What\'s new opens the Browse hub on Episodes', async ({ page }) => {
  await page.goto('/')
  const section = page.getByRole('heading', { name: "What's new" }).locator('xpath=ancestor::section[1]')
  await section.getByRole('link', { name: /Browse all/ }).click()
  await expect(page).toHaveURL(/\/browse\?tab=episodes/)
})

test('UXS-012 §Section state: a failed What\'s new says so and Retry recovers it', async ({ page }) => {
  // EVERY What's new request fails until Retry is pressed — more than one Home consumer may ask, so
  // "fail the first" made which section saw the failure a race.
  // A 500, NOT a 503: the app reads 503 as "the server is degraded" (services/api.ts) and shows the
  // app-wide "couldn't reach the server" banner instead — a different state, which won the race
  // against the section's own error intermittently (desktop, 2026-10-06).
  let failing = true
  await page.route('**/api/app/whats-new**', async (route) => {
    if (failing) {
      await route.fulfill({ status: 500, body: 'section failed' })
      return
    }
    await route.continue()
  })
  // Sign-in lands on Home, which loads What's new and CACHES it (offline-first, per account). With
  // that cache present a failed load correctly shows the cached rows under "Showing what you had
  // last time" — not the section error this test is about — so it passed or failed on whether the
  // sign-in's own load had finished (2026-10-06). Drop that one cache entry first.
  await page.evaluate(() => {
    for (const k of Object.keys(localStorage)) {
      if (k.endsWith('.home.whatsnew.v2')) localStorage.removeItem(k)
    }
  })
  await page.goto('/')
  const error = page.getByTestId('section-error').first()
  await expect(error).toBeVisible()
  failing = false
  await page.getByTestId('section-retry').first().click()
  await expect(page.getByRole('heading', { name: "What's new" })).toBeVisible()
  await expect(page.getByTestId('section-error')).toHaveCount(0)
})
