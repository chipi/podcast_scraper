import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Entity pages — topic, theme, storyline, person (operator 2026-10-05, UXS-011 §Entity pages).
 *
 * Real API + committed corpus. Layout is measured, never eyeballed, and always from ONE layout frame:
 * two sequential boundingBox() reads let a section settling above them fake an overlap.
 */
const TOPIC = `/topic/${encodeURIComponent('topic:risk-management')}`
const THEME = `/theme/${encodeURIComponent('tc:safety-practices')}`
const STORYLINE = `/storyline/${encodeURIComponent('topic:risk-management')}`
const NORA = `/person/${encodeURIComponent('person:nora')}`
const MAYA = `/person/${encodeURIComponent('person:maya')}`

type Box = { x: number; y: number; width: number; height: number }
/** Boxes of `ids` inside `view`, all read in one frame; null for an id that is not rendered. */
async function boxes(page: Page, view: string, ids: string[]): Promise<Record<string, Box | null>> {
  return page.evaluate(
    ([view, ids]) => {
      const v = document.querySelector(`[data-testid="${view}"]`)!
      const out: Record<string, { x: number; y: number; width: number; height: number } | null> = {}
      for (const id of ids) {
        const e = v.querySelector(id.startsWith('[') || id.includes(' ') ? id : `[data-testid="${id}"]`)
        const r = e?.getBoundingClientRect()
        out[id] = r && r.height ? { x: r.x, y: r.y, width: r.width, height: r.height } : null
      }
      return out
    },
    [view, ids] as const,
  )
}

test.describe('topic page', () => {
  test('similar topics are an inline label; theme and storyline are link cards that name their kind', async ({
    page,
  }, testInfo) => {
    await signInIsolated(page, 'entity-topic-sections', testInfo)
    await page.goto(TOPIC)
    const v = page.getByTestId('topic-view')
    const similar = v.getByTestId('ec-similar-topics')
    await expect(similar).toBeVisible()
    // No heading over the similar topics: the count is a kicker label in the pill row.
    await expect(similar.locator('h2, h3')).toHaveCount(0)
    await expect(similar.getByTestId('ec-similar-label')).toHaveText(/^\d+ similar topics?$/)
    // No "Part of a theme / storyline" headings: the link card is the section, captioned by kind.
    await expect(v.getByText('Part of a theme')).toHaveCount(0)
    await expect(v.getByText('Part of a storyline')).toHaveCount(0)
    await expect(v.getByTestId('ec-theme-kind')).toHaveText(/^Theme · \d+ topics$/i)
    await expect(v.getByTestId('ec-storyline-kind')).toHaveText(/^Storyline · \d+ topics$/i)
    await expect(v.getByTestId('topic-perspectives')).toContainText('perspectives on risk management')
  })

  test('desktop: the two charts share a row at the same height; Top voices alone on the right', async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await signInIsolated(page, 'entity-topic-desktop', testInfo)
    await page.goto(TOPIC)
    const v = page.getByTestId('topic-view')
    await expect(v.getByTestId('tca-bars')).toBeVisible()
    await expect(v.getByTestId('ec-topic-activity-head')).toContainText('Discussed over time')
    const b = await boxes(page, 'topic-view', [
      '[data-testid="ec-topic-activity"] svg',
      'tca-bars',
      'ec-similar-topics',
      'ec-theme',
      'ec-storyline',
      'ec-top-voices',
    ])
    const spark = b['[data-testid="ec-topic-activity"] svg']!
    const bars = b['tca-bars']!
    expect(Math.abs(spark.y - bars.y), 'charts start on one line').toBeLessThanOrEqual(2)
    expect(Math.abs(spark.height - bars.height), 'charts are one height').toBeLessThanOrEqual(2)
    expect(bars.x, 'arc to the right').toBeGreaterThan(spark.x + spark.width)
    // Left column: similar → theme → storyline, stacked. Right column: Top voices alone.
    const [sim, th, sl, tv] = [b['ec-similar-topics']!, b['ec-theme']!, b['ec-storyline']!, b['ec-top-voices']!]
    expect(th.y).toBeGreaterThanOrEqual(sim.y + sim.height)
    expect(sl.y).toBeGreaterThanOrEqual(th.y + th.height)
    expect(Math.abs(tv.y - sim.y), 'Top voices starts beside similar topics').toBeLessThanOrEqual(2)
    expect(tv.x).toBeGreaterThan(sl.x + sl.width)
  })

  test('phone: one column — badge/caption layout, and Top voices follows the storyline link', async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 412, height: 915 })
    await signInIsolated(page, 'entity-topic-phone', testInfo)
    await page.goto(TOPIC)
    const v = page.getByTestId('topic-view')
    await expect(v.getByTestId('ec-top-voices')).toBeVisible()
    await expect(v.getByTestId('ec-topic-activity-head')).toBeHidden()
    await expect(v.getByTestId('ec-topic-activity').locator('figcaption')).toBeVisible()
    const b = await boxes(page, 'topic-view', ['ec-storyline', 'ec-top-voices'])
    expect(b['ec-top-voices']!.y).toBeGreaterThanOrEqual(b['ec-storyline']!.y + b['ec-storyline']!.height)
  })
})

test.describe('theme and storyline pages', () => {
  for (const [url, view, kicker, topics] of [
    [THEME, 'theme-view', 'Theme', 'theme-topics'],
    [STORYLINE, 'storyline-view', 'Storyline', 'storyline-topics'],
  ] as const) {
    test(`${kicker}: singular label; members beside Top voices; episodes close the page at full width`, async ({
      page,
    }, testInfo) => {
      await page.setViewportSize({ width: 1440, height: 900 })
      await signInIsolated(page, `entity-${view}`, testInfo)
      await page.goto(url)
      const v = page.getByTestId(view)
      await expect(v.getByTestId(topics)).toBeVisible()
      // The page label names ONE theme / storyline, not the list ("THEMES").
      await expect(v.getByText(kicker, { exact: true }).first()).toBeVisible()
      await expect(v.getByText(`${kicker}s`, { exact: true })).toHaveCount(0)
      const episodes = view === 'theme-view' ? 'theme-episodes' : 'storyline-episodes'
      await expect(v.getByTestId(episodes)).toBeVisible()
      // "What's said" is its own request — measure only once it has rendered.
      await expect(v.getByTestId('topic-perspectives')).toBeVisible()
      await expect(v.getByTestId('ec-top-voices')).toBeVisible()
      const b = await boxes(page, view, [topics, 'ec-top-voices', 'topic-perspectives', episodes])
      const [members, voices, said, eps] = [b[topics]!, b['ec-top-voices']!, b['topic-perspectives']!, b[episodes]!]
      expect(Math.abs(voices.y - members.y), 'Top voices beside the members').toBeLessThanOrEqual(2)
      expect(said.y, "what's said after the pair").toBeGreaterThanOrEqual(Math.max(members.y + members.height, voices.y + voices.height))
      expect(eps.y, 'episodes after what is said').toBeGreaterThanOrEqual(said.y + said.height)
      // Full width: the episode list spans both columns.
      expect(eps.width).toBeGreaterThan(members.width + voices.width)
    })
  }
})

test.describe('person page', () => {
  test('order, the mixed related group, and "Who agrees with {name}"', async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 412, height: 915 })
    await signInIsolated(page, 'entity-person', testInfo)
    await page.goto(NORA)
    const v = page.getByTestId('person-view')
    await expect(v.getByTestId('es-consensus')).toBeVisible()
    // Order down the page.
    const order = ['es-coappears', 'ec-related-person', 'ec-person-related', 'ec-search-library', 'es-consensus']
    const b = await boxes(page, 'person-view', order)
    for (let i = 1; i < order.length; i++) {
      expect(b[order[i]]!.y, `${order[i]} below ${order[i - 1]}`).toBeGreaterThan(b[order[i - 1]]!.y)
    }
    // Related topics: themes and storylines first, every pill naming its kind.
    const related = v.getByTestId('ec-person-related')
    const kinds = await related.locator('button').evaluateAll((els) =>
      els.map((e) => (e.querySelector('span')?.textContent ?? '').trim().toLowerCase()),
    )
    expect(kinds.length).toBeGreaterThan(0)
    const firstTopic = kinds.indexOf('topic')
    expect(kinds.slice(0, firstTopic).every((k) => k === 'theme' || k === 'storyline')).toBe(true)
    expect(kinds.slice(firstTopic).every((k) => k === 'topic')).toBe(true)
    // "Who agrees with Nora": each row names the other person ONCE, on the agreeing line.
    await expect(v.getByTestId('es-consensus')).toContainText('Who agrees with Nora')
    const row = v.getByTestId('es-consensus-row').first()
    const other = await row.getByTestId('es-consensus-other').locator('button').innerText()
    await expect(row.getByTestId('es-consensus-other')).toContainText(`${other} agrees:`)
    expect((await row.innerText()).split(other).length - 1).toBe(1)
  })

  test('agreements page five at a time', async ({ page }, testInfo) => {
    await signInIsolated(page, 'entity-person-paging', testInfo)
    await page.goto(NORA)
    const v = page.getByTestId('person-view')
    const rows = v.getByTestId('es-consensus-row')
    await expect(rows).toHaveCount(5)
    const more = v.getByTestId('es-consensus-more')
    await expect(more).toHaveText(/^Show more \(\d+\)$/)
    await more.click()
    await expect(more).toHaveText('Show less')
    expect(await rows.count()).toBeGreaterThan(5)
    await more.click()
    await expect(rows).toHaveCount(5)
  })

  test('a theme pill opens the theme on top, as a sheet', async ({ page }, testInfo) => {
    await signInIsolated(page, 'entity-person-theme', testInfo)
    await page.goto(NORA)
    await page.getByTestId('ec-person-related-theme').first().click()
    await expect(page.getByTestId('theme-card')).toBeVisible()
    await expect(page.getByTestId('person-view')).toBeVisible()
  })

  test('"Host of A and B" keeps its spaces', async ({ page }, testInfo) => {
    await signInIsolated(page, 'entity-person-host', testInfo)
    await page.goto(MAYA)
    const host = page.getByTestId('person-view').getByTestId('ec-host-shows').first()
    await expect(host).toBeVisible()
    await expect(host).toHaveText(/^Host of .+ and .+$/)
    expect(await host.innerText()).not.toMatch(/\Sand\S/)
  })
})

test('entity sheets are 768px wide on desktop', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await signInIsolated(page, 'entity-sheet-width', testInfo)
  await page.goto(TOPIC)
  await page.getByTestId('topic-view').getByTestId('ec-storyline-link').click()
  const sheet = page.getByTestId('storyline-card')
  await expect(sheet).toBeVisible()
  const width = await page.locator('.lp-sheet').last().evaluate((e) => e.getBoundingClientRect().width)
  expect(Math.round(width)).toBe(768)
})
