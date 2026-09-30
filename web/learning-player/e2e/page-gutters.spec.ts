import { expect, test, type Page } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Entity PAGES share one gutter (operator 2026-09-30).
 *
 * A topic opened from Home landed on /topic/:id with visibly more side padding than a storyline
 * page opened from inside it: TopicView wrapped the card in `px-4` AND the card (EntityCardBody)
 * padded its own header and body with `px-4`, so the topic page had 32px a side against the
 * storyline page's 16px. PersonView had the same double gutter.
 *
 * Measured, not eyeballed: the title's inset from the page container must be the same on all three.
 */
async function titleInset(page: Page, container: string, title: string): Promise<number> {
  const c = page.getByTestId(container)
  await expect(c).toBeVisible()
  const h = c.locator(title).first()
  await expect(h).toBeVisible()
  const [cb, hb] = [await c.boundingBox(), await h.boundingBox()]
  return Math.round(hb!.x - cb!.x)
}

test('topic, person and storyline pages inset their content by the same gutter', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'page-gutters', testInfo)

  await page.goto(`/storyline/${encodeURIComponent('topic:risk-management')}`)
  const storyline = await titleInset(page, 'storyline-view', 'h1')

  await page.goto(`/topic/${encodeURIComponent('topic:risk-management')}`)
  const topic = await titleInset(page, 'topic-view', 'h2')

  await page.goto(`/person/${encodeURIComponent('person:jack-clark')}`)
  const person = await titleInset(page, 'person-view', 'h2')

  // One page gutter (px-4). Within 2px, because a heading's box can sit a pixel inside its padding
  // edge; the defect this guards was a 16px difference.
  expect(Math.abs(storyline - 16), `storyline page gutter ${storyline}`).toBeLessThanOrEqual(2)
  expect(Math.abs(topic - storyline), `topic page gutter ${topic} vs ${storyline} (was 32)`).toBeLessThanOrEqual(2)
  expect(Math.abs(person - storyline), `person page gutter ${person} vs ${storyline} (was 32)`).toBeLessThanOrEqual(2)
})
