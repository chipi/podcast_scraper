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
 *
 * Since 2026-10-05 that inset is ZERO (operator: one page width, nothing narrows a page further).
 * The app shell's gutter is the only gutter; the card renders `flush` on these routes, so a topic,
 * person or storyline title starts at the same left edge as Home's.
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

  await page.goto(`/person/${encodeURIComponent('person:nora')}`)
  const person = await titleInset(page, 'person-view', 'h2')

  // No gutter inside the page. Within 2px, because a heading's box can sit a pixel inside its
  // padding edge; the defects this guards were 16px differences.
  expect(Math.abs(storyline), `storyline page gutter ${storyline}`).toBeLessThanOrEqual(2)
  expect(Math.abs(topic - storyline), `topic page gutter ${topic} vs ${storyline} (was 32)`).toBeLessThanOrEqual(2)
  expect(Math.abs(person - storyline), `person page gutter ${person} vs ${storyline} (was 32)`).toBeLessThanOrEqual(2)
})

/**
 * Home and Discover render the SAME Trends component, so it must sit at the same place and width
 * on both (operator 2026-10-05). Discover wrapped it in an extra `px-4`, which made it 32px
 * narrower than on Home — enough that the fourth kind pill (Themes) collided with the sort switch.
 */
test('Trends sits at the same inset and width on Home and Discover', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trends-gutter', testInfo)
  const box = async (url: string) => {
    await page.goto(url)
    const el = page.getByTestId('discovery-explorer')
    await expect(el).toBeVisible()
    return (await el.boundingBox())!
  }
  const home = await box('/')
  const discover = await box('/browse')
  expect(Math.round(discover.x)).toBe(Math.round(home.x))
  expect(Math.round(discover.width)).toBe(Math.round(home.width))
})

/** Four kind pills and the two switches share one row, down to a 375px phone, on both screens. */
test('the Trends kind pills clear the switches at 375px, on Home and Discover', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile-chrome', 'a phone-width layout')
  await page.setViewportSize({ width: 375, height: 800 })
  await signInIsolated(page, 'trends-375', testInfo)
  for (const url of ['/', '/browse']) {
    await page.goto(url)
    const people = (await page.getByTestId('discovery-tab-person').boundingBox())!
    const sort = (await page.getByTestId('discovery-sort').boundingBox())!
    const scope = (await page.getByTestId('home-trending-scope').boundingBox())!
    expect(people.x + people.width, `${url}: People runs into the sort switch`).toBeLessThanOrEqual(sort.x)
    expect(scope.x + scope.width, `${url}: the scope switch leaves the screen`).toBeLessThanOrEqual(375)
  }
})

/**
 * The two exceptions to the one page width (operator 2026-10-05), both centred at every width:
 * sign-in (448px, like most sites — at the full width its name field ran ~1000px) and the account
 * pages behind the avatar (42rem = 672px — settings rows put a label ~1100px from its switch).
 */
async function centred(page: Page, url: string, testid: string, max: number) {
  await page.goto(url)
  const box = (await page.getByTestId(testid).boundingBox())!
  expect(box.width, url).toBeLessThanOrEqual(max)
  expect(Math.abs(box.x - (1440 - (box.x + box.width))), `${url} centred`).toBeLessThanOrEqual(2)
}

test('sign-in is a centred column on desktop', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await centred(page, '/login', 'login-view', 448)
  await centred(page, '/login?mode=signup', 'login-view', 448)
})

test('the account pages behind the avatar are a centred column on desktop', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await signInIsolated(page, 'account-pages-centred', testInfo)
  await centred(page, '/profile', 'profile-view', 672)
  await centred(page, '/settings', 'settings-view', 672)
  await centred(page, '/about/privacy', 'about-page', 672)
  await centred(page, '/account/delete', 'delete-account-view', 672)
})
