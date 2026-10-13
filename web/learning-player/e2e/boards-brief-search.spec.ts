import { expect, test, type Page } from '@playwright/test'
import { navTo, signInIsolated } from './helpers'

/**
 * A board's Brief, its short listen, and boards in search (RFC-119 amendment 2026-10-12; #2340,
 * #2341), against the real api and the fixture corpus.
 *
 * The board is built through the API (the add-to-board sheet has its own spec, `collections.spec`);
 * what is under test here is what the board then SAYS and PLAYS, and that search finds it. Every
 * name carries this run's id, and assertions are scoped to it: `signInIsolated` keeps the account
 * between runs, so boards accumulate.
 */

async function threeEpisodesOfOneShow(page: Page): Promise<string[]> {
  const res = await page.request.get('/api/app/episodes?page_size=100')
  const items = (await res.json()).items as Array<{ slug: string; feed_id: string }>
  const byShow = new Map<string, string[]>()
  for (const e of items) byShow.set(e.feed_id, [...(byShow.get(e.feed_id) ?? []), e.slug])
  const three = [...byShow.values()].find((slugs) => slugs.length >= 3)
  expect(three, 'the fixture has a show with three episodes').toBeTruthy()
  return three!.slice(0, 3)
}

async function makeBoard(page: Page, name: string, items: Array<Record<string, string>>): Promise<string> {
  const created = await page.request.post('/api/app/collections', { data: { name } })
  expect(created.ok(), `creating ${name}: ${created.status()}`).toBeTruthy()
  const id = (await created.json()).id as string
  for (const item of items) {
    const added = await page.request.post(`/api/app/collections/${id}/items`, { data: item })
    expect(added.ok(), `adding to ${name}: ${added.status()}`).toBeTruthy()
  }
  return id
}

test('a board opens its Brief from its own button, and plays its own moments', async ({ page }, testInfo) => {
  await signInIsolated(page, 'board-brief', testInfo)
  const run = Date.now().toString(36)
  const name = `Brief ${run}`
  const slugs = await threeEpisodesOfOneShow(page)
  const id = await makeBoard(
    page,
    name,
    slugs.map((ref) => ({ kind: 'episode', ref })),
  )

  await navTo(page, 'library')
  await page.getByTestId('library-tab-collections').click()
  const row = page.locator('[data-board-row]').filter({ hasText: name })
  await row.getByTestId('collection-open').click()
  await expect(page.getByTestId('collection-items')).toBeVisible()
  // Not shown by default (operator 2026-10-12): the Brief opens from the board's own button.
  const brief = page.getByTestId('board-brief')
  await expect(brief).toBeHidden()
  await row.getByTestId('collection-brief').click()
  await expect(brief).toBeVisible()
  await expect(brief.getByTestId('board-brief-holds')).toHaveText('3 episodes')
  // One key point per episode, each opening its episode.
  const points = brief.getByTestId('board-brief-point')
  await expect(points).toHaveCount(3)
  for (const [i, slug] of slugs.entries()) {
    await expect(points.nth(i)).toHaveAttribute('href', new RegExp(`^/episode/${slug}(\\?t=\\d+)?$`))
  }
  // Its voices, and what its episodes are about.
  await expect(brief.getByTestId('ec-top-voice').first()).toBeVisible()
  await expect(brief.getByTestId('board-brief-subject').first()).toBeVisible()

  // The board's own short listen, through the same player as a subject's.
  const door = brief.getByTestId('board-brief-play')
  await expect(door).toBeVisible()
  await door.click()
  await expect(brief).toBeHidden()
  await expect(page).toHaveURL(new RegExp(`/listen/topic/board:${id}\\?minutes=15$`))
  await expect(page.getByTestId('topic-reel-title')).toHaveText(name)
  await expect(page.getByTestId('topic-reel-length')).toContainText('3 clips')
})

test('search finds your boards by name and by what is saved on them', async ({ page }, testInfo) => {
  await signInIsolated(page, 'board-search', testInfo)
  const run = Date.now().toString(36)
  const byName = await makeBoard(page, `Zq${run} lab`, [])
  const byLink = await makeBoard(page, `Reading ${run}`, [
    { kind: 'link', ref: `https://example.org/${run}`, title: `Notes yx${run}` },
  ])

  await page.goto(`/search?q=${encodeURIComponent(`zq${run}`)}&scope=all`)
  const section = page.getByTestId('search-board-matches')
  await expect(section).toBeVisible()
  const named = section.getByTestId('search-board')
  await expect(named).toHaveCount(1)
  await expect(named).toHaveAttribute('href', `/library?tab=collections&board=${byName}`)
  await expect(named.getByTestId('search-board-why')).toHaveText('Name matches')

  await page.goto(`/search?q=${encodeURIComponent(`yx${run} notes`)}&scope=all`)
  const linked = page.getByTestId('search-board-matches').getByTestId('search-board')
  await expect(linked).toHaveCount(1)
  await expect(linked.getByTestId('search-board-why')).toHaveText(`Link: Notes yx${run}`)
  // It opens the board, open, in Library.
  await linked.click()
  await expect(page).toHaveURL(new RegExp(`board=${byLink}`))
  await expect(page.getByTestId('collection-items')).toBeVisible()
})
