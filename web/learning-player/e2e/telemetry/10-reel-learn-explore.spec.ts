import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import { showEveryonesTrends } from '../helpers'
import './settle'

/**
 * Telemetry for what the Discover arc added late (operator 2026-10-11: "real Umami telemetry for all
 * new things"): the reel, "Learn this week", Explore's taste, Trends paging and the list filters —
 * each event reaching the DEV Umami site and ACCEPTED by it, with the properties the dashboards will
 * read, and never which episode or subject.
 */

test('a reel reports how it was opened, and what the listener did inside it', async ({ page }) => {
  const sink = attachSink(page)
  await page.goto('/api/app/auth/login?as=telemetry-reel')
  await page.waitForLoadState('networkidle')
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('reel-topic-chip').filter({ hasText: 'Managing risk across domains' }).click()
  const open = await sink.waitForEvent('reel_open')
  expect(open.data).toMatchObject({ source: 'discover_row', kind: 'storyline' })

  await page.getByTestId('topic-reel-play').click()
  const start = await sink.waitForEvent('reel_action')
  expect(start.data).toMatchObject({ action: 'start', kind: 'storyline', clips: '21+' })
  await expect(page.getByTestId('topic-reel-current')).toBeVisible()
  await page.getByTestId('topic-reel-next').click()
  await expect.poll(() => sink.byName('reel_action').map((e) => e.data?.action)).toContain('next')
  const next = sink.byName('reel_action').find((e) => e.data?.action === 'next')!
  expect(next.data?.clip).toBe('1') // skipped on from the first clip
  await page.getByTestId('topic-reel-keep').click()
  await expect.poll(() => sink.byName('reel_action').map((e) => e.data?.action)).toContain('keep')

  // From a subject page too.
  await page.goto(`/theme/${encodeURIComponent('tc:reliability')}`)
  await page.getByTestId('reel-door').click()
  await expect
    .poll(() => sink.byName('reel_open').map((e) => `${e.data?.source}:${e.data?.kind}`))
    .toContain('subject_page:theme')

  expect(JSON.stringify(sink.byName('reel_open').map((e) => e.raw))).not.toMatch(/thc:|tc:|topic:|episode\//)
  expect(sink.rejectedBeacons(), 'Umami accepted every beacon').toEqual([])
})

test('Learn this week reports where a path started and a step played', async ({ page }) => {
  const sink = attachSink(page)
  await page.goto('/api/app/auth/login?as=telemetry-learn')
  await page.waitForLoadState('networkidle')
  await page.goto('/browse')
  await showEveryonesTrends(page)
  await page.getByTestId('learn-subject').filter({ hasText: 'Managing risk across domains' }).click()
  const start = await sink.waitForEvent('learn_path')
  expect(start.data).toMatchObject({ action: 'start', kind: 'storyline', source: 'discover' })

  await page.getByTestId('learn-this-week').getByTestId('learn-step-play').click()
  await expect.poll(() => sink.byName('learn_path').map((e) => e.data?.action)).toContain('step_play')
  const play = sink.byName('learn_path').find((e) => e.data?.action === 'step_play')!
  expect(play.data).toMatchObject({ kind: 'storyline', day: 1 })
  expect(sink.rejectedBeacons(), 'Umami accepted every beacon').toEqual([])
})

test("Explore reports Trends paging, the catalog taste's See all, and the list filters", async ({ page }) => {
  const sink = attachSink(page)
  await page.goto('/api/app/auth/login?as=telemetry-explore')
  await page.waitForLoadState('networkidle')
  await page.goto('/explore')
  await showEveryonesTrends(page)
  await page.getByTestId('browse-discovery').getByTestId('show-all-toggle').click()
  const more = await sink.waitForEvent('trends_show_more')
  expect(more.data).toMatchObject({ kind: 'topic', sort: 'rising' })
  expect(['1', '2-5', '6-20', '21+']).toContain(String(more.data?.shown))

  await page.getByTestId('browse-tab-shows').click()
  await page.getByTestId('explore-see-all-shows').click()
  const all = await sink.waitForEvent('explore_see_all')
  expect(all.data).toEqual({ list: 'shows' })

  await expect(page).toHaveURL(/\/shows$/)
  await page.getByTestId('show-browse-from').click()
  await page.getByTestId('show-browse-from-opt-mine').click()
  const filter = await sink.waitForEvent('catalog_filter')
  expect(filter.data).toEqual({ list: 'shows', value: 'mine' })
  expect(sink.rejectedBeacons(), 'Umami accepted every beacon').toEqual([])
})
