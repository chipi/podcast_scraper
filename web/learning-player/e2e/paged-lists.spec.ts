import { expect, test, type APIRequestContext, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Long lists page (operator 2026-10-05, UXS-011 §Long lists page) — every list here grows without a
 * server limit, so each is SEEDED past its page through the real API (the signed-in session's own
 * cookies), then walked: the first page, "Show more (N)" for the rest, "Show less" back to the
 * first page. One mechanism throughout (useCappedSections + ShowAllToggle), so one label shape.
 */
const TOPIC_ID = 'topic:risk-management'

async function ok(res: Awaited<ReturnType<APIRequestContext['post']>>, what: string) {
  expect(res.ok(), `${what}: ${res.status()}`).toBeTruthy()
  return res.json()
}

async function seedNotes(page: Page, target: string, targetId: string, n: number, word: string) {
  for (let i = 0; i < n; i++) {
    await ok(
      await page.request.post('/api/app/notes', { data: { target, target_id: targetId, text: `${word} note ${i}` } }),
      `seeding note ${i}`,
    )
  }
}

/** Walk a paged list: `first` shown, the control reveals the rest, then folds back. */
async function walk(page: Page, items: ReturnType<Page['getByTestId']>, more: ReturnType<Page['getByTestId']>, first: number, total: number) {
  await expect(items).toHaveCount(first)
  await expect(more).toHaveText(`Show more (${Math.min(total - first, total)})`)
  while ((await more.innerText()) !== 'Show less') await more.click()
  await expect(items).toHaveCount(total)
  await more.click()
  await expect(items).toHaveCount(first)
}

test('notes in a notes box: newest first, five at a time', async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-notes-${Date.now()}`, testInfo)
  await seedNotes(page, 'topic', TOPIC_ID, 7, 'pagenote')
  await page.goto(`/topic/${encodeURIComponent(TOPIC_ID)}`)
  const box = page.getByTestId('topic-view').getByTestId('note-composer')
  const items = box.getByTestId('note-item')
  await expect(items.first()).toContainText('pagenote note 6')
  await walk(page, items, box.getByTestId('notes-more'), 5, 7)
})

test("Search's \"Your notes\" pages five at a time", async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-search-notes-${Date.now()}`, testInfo)
  await seedNotes(page, 'topic', TOPIC_ID, 7, 'zqsearchpage')
  await page.goto('/search?q=zqsearchpage')
  const section = page.getByTestId('search-note-matches')
  await expect(section).toBeVisible()
  await walk(page, section.getByTestId('search-note'), section.getByTestId('search-notes-more'), 5, 7)
})

test("a highlight's notes page five at a time", async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-highlight-notes-${Date.now()}`, testInfo)
  const eps = (await (await page.request.get('/api/app/episodes?page_size=5')).json()) as { items: { slug: string }[] }
  const hl = await ok(
    await page.request.post('/api/app/highlights', { data: { episode_slug: eps.items[0].slug, kind: 'moment', start_ms: 1000 } }),
    'seeding a highlight',
  )
  await seedNotes(page, 'highlight', hl.id, 7, 'hlnote')
  await page.goto('/library?tab=saved')
  const more = page.getByTestId('highlight-notes-more')
  await expect(more).toBeVisible()
  const notes = page.getByTestId('highlight-note')
  await expect(notes.first()).toContainText('hlnote note 6')
  await walk(page, notes, more, 5, 7)
})

test("Library's saved topics page five at a time", async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-saved-topics-${Date.now()}`, testInfo)
  const hits = (await (await page.request.get('/api/app/interests/search?kind=topic&q=a&limit=50')).json()) as {
    items: { id: string; label: string }[]
  }
  const topics = hits.items.slice(0, 7)
  expect(topics.length, 'the corpus has seven topics to save').toBe(7)
  for (const tp of topics) {
    const res = await page.request.put('/api/app/favorites', { data: { kind: 'topic', ref: tp.id, label: tp.label } })
    expect(res.ok(), `saving ${tp.id}: ${res.status()}`).toBeTruthy()
  }
  await page.goto('/library?tab=saved')
  const section = page.locator('section', { has: page.getByRole('heading', { name: /^Topics/ }) })
  await expect(section.getByTestId('saved-entity')).toHaveCount(5)
  const more = section.getByTestId('show-all-toggle')
  await expect(more).toHaveText('Show more (2)')
  await more.click()
  await expect(section.getByTestId('saved-entity')).toHaveCount(7)
  await expect(more).toHaveText('Show less')
})

test("an open board's items page ten at a time", async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-board-items-${Date.now()}`, testInfo)
  const board = await ok(await page.request.post('/api/app/collections', { data: { name: 'Paging board' } }), 'seeding a board')
  const eps = (await (await page.request.get('/api/app/episodes?page_size=12')).json()) as { items: { slug: string; title: string }[] }
  expect(eps.items.length).toBe(12)
  for (const e of eps.items) {
    await ok(
      await page.request.post(`/api/app/collections/${encodeURIComponent(board.id)}/items`, {
        data: { kind: 'episode', ref: e.slug, title: e.title },
      }),
      `adding ${e.slug}`,
    )
  }
  await page.goto('/library?tab=collections')
  await page.getByTestId('collection-open').filter({ hasText: 'Paging board' }).first().click()
  await walk(page, page.getByTestId('collection-item'), page.getByTestId('collection-items-more'), 10, 12)
})

test('followed interests page ten at a time per kind', async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-interests-${Date.now()}`, testInfo)
  const hits = (await (await page.request.get('/api/app/interests/search?kind=topic&q=a&limit=50')).json()) as {
    items: { id: string }[]
  }
  const ids = hits.items.slice(0, 12).map((h) => h.id)
  expect(ids.length).toBe(12)
  const res = await page.request.put('/api/app/interests', { data: { items: ids } })
  expect(res.ok(), `following 12 topics: ${res.status()}`).toBeTruthy()
  await page.goto('/profile?tab=interests')
  const section = page.getByTestId('interests-section-topic')
  await walk(page, section.getByTestId('interest-following-topic'), section.getByTestId('interest-following-more-topic'), 10, 12)
})

test('All episodes reveals matches 20 at a time while a sort is active', async ({ page }, testInfo) => {
  await signInIsolated(page, `paged-catalog-${Date.now()}`, testInfo)
  const all = (await (await page.request.get('/api/app/episodes?page_size=50')).json()) as { total: number }
  test.skip(all.total <= 20, `the corpus has ${all.total} episodes — needs more than 20 to page`)
  await page.goto('/catalog')
  await expect(page.getByTestId('episode-card').first()).toBeVisible()
  await page.getByTestId('list-toolbar-sort').click()
  await page.getByTestId('list-toolbar-sort-opt-oldest').click()
  // The sort fetches every page, but renders the first 20 behind "Load more".
  await expect(page.getByTestId('episode-card')).toHaveCount(20)
  await page.getByTestId('catalog-load-more').click()
  await expect(page.getByTestId('episode-card')).toHaveCount(Math.min(40, all.total))
})
