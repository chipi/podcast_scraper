import { expect, test, type Page } from '@playwright/test'
import { signInIsolated, showEveryonesTrends } from './helpers'

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
 * Trends sits at the page's own inset and width (operator 2026-10-05): Discover once wrapped it in an
 * extra `px-4`, which made it 32px narrower than the search above it — enough that the fourth kind
 * pill (Themes) collided with the sort switch. Since Trends left Home (2026-10-07) the reference is
 * the search section beside it on the same page.
 */
test('Trends sits at the same inset and width as the search on Discover', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trends-gutter', testInfo)
  await page.goto('/browse')
  await showEveryonesTrends(page)
  const trends = page.getByTestId('discovery-explorer')
  const search = page.getByTestId('browse-search-section')
  await expect(trends).toBeVisible()
  await expect(search).toBeVisible()
  const t = (await trends.boundingBox())!
  const s = (await search.boundingBox())!
  expect(Math.round(t.x)).toBe(Math.round(s.x))
  expect(Math.round(t.width)).toBe(Math.round(s.width))
})

/**
 * Four kind pills and the two switches share one row on a phone — and never
 * overlap. The pills render in the device's system font, so their width is the OS's: macOS left 6px
 * of slack at 375px while CI's Linux font ran People 13px into the sort switch (2026-10-06). When
 * the pills do not fit, the strip stops at the switches and scrolls; 360px is a common Android width.
 */
for (const width of [360, 375]) {
  test(`the Trends kind pills stop at the switches at ${width}px on Discover`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'mobile-chrome', 'a phone-width layout')
    await page.setViewportSize({ width, height: 800 })
    await signInIsolated(page, `trends-${width}`, testInfo)
    for (const url of ['/browse']) {
      await page.goto(url)
      await expect(page.getByTestId('discovery-tab-person')).toBeVisible()
      // One read of every box, so nothing reflows between them. A pill past the sort switch is
      // only acceptable inside a strip that clips and scrolls — never drawn over the switch.
      const m = await page.evaluate(`(() => {
        const r = (id) => document.querySelector('[data-testid="' + id + '"]').getBoundingClientRect()
        const strip = document.querySelector('[data-testid="discovery-tab-person"]').parentElement
        const sortX = r('discovery-sort').left
        const pills = Array.prototype.slice.call(strip.children)
        const pastSwitch = pills.filter((el) => el.getBoundingClientRect().right > sortX).length
        return {
          pastSwitch,
          stripClips: getComputedStyle(strip).overflowX !== 'visible',
          stripRight: strip.getBoundingClientRect().right,
          sortX,
          scopeRight: r('home-trending-scope').right,
        }
      })()`) as { pastSwitch: number; stripClips: boolean; stripRight: number; sortX: number; scopeRight: number }
      expect(m.stripRight, `${url}: the kind strip runs into the sort switch`).toBeLessThanOrEqual(m.sortX)
      if (m.pastSwitch > 0) expect(m.stripClips, `${url}: ${m.pastSwitch} pill(s) drawn over the sort switch`).toBe(true)
      expect(m.scopeRight, `${url}: the scope switch leaves the screen`).toBeLessThanOrEqual(width)
      // Every kind stays reachable — People is the last pill, the one a narrow row hides.
      await page.getByTestId('discovery-tab-person').click()
      await expect(page.getByTestId('discovery-tab-person')).toHaveAttribute('aria-selected', 'true')
    }
  })
}

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

/**
 * Desktop entity pages put sections side by side, half the width each (operator 2026-10-05): on
 * the topic page similar topics, "Part of a theme" and "Part of a storyline" stack in the left
 * column with Top voices alone on the right; on theme and storyline pages the member topics sit
 * beside Top voices. On a phone everything stacks.
 */
for (const viewport of [
  { width: 1440, height: 900, side: true },
  { width: 412, height: 915, side: false },
]) {
  test(`entity sections pair up only on desktop (${viewport.width}px)`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    await signInIsolated(page, `entity-pairs-${viewport.width}`, testInfo)
    // Scoped to the page being measured: right after a navigation the page being left can still be
    // on screen in its exit transition, carrying its own copy of a section (Top voices).
    const pair = async (url: string, view: string, a: string, b: string) => {
      await page.goto(url)
      const v = page.getByTestId(view)
      const [ba, bb] = [v.getByTestId(a).first(), v.getByTestId(b).first()]
      await expect(ba).toBeVisible()
      await expect(bb).toBeVisible()
      // Both boxes from ONE layout: measured one after the other, a section above them that settles
      // in between (async content) moves the second box against the first and fakes an overlap.
      const [ra, rb] = await page.evaluate(
        ([view, a, b]) => {
          const v = document.querySelector(`[data-testid="${view}"]`)!
          const box = (id: string) => {
            const r = v.querySelector(`[data-testid="${id}"]`)!.getBoundingClientRect()
            return { x: r.x, y: r.y, width: r.width, height: r.height }
          }
          return [box(a), box(b)]
        },
        [view, a, b] as const,
      )
      if (viewport.side) {
        expect(Math.abs(ra.y - rb.y), `${url}: ${a} and ${b} share a row`).toBeLessThanOrEqual(2)
        expect(rb.x, `${url}: ${b} sits to the right`).toBeGreaterThan(ra.x + ra.width)
      } else {
        expect(rb.y, `${url}: ${b} stacks below ${a}`).toBeGreaterThanOrEqual(ra.y + ra.height)
      }
    }
    const topic = `/topic/${encodeURIComponent('topic:risk-management')}`
    await pair(topic, 'topic-view', 'ec-similar-topics', 'ec-top-voices')
    // The storyline link stacks under the theme link at every width (both in the left column).
    await page.goto(topic)
    const tv = page.getByTestId('topic-view')
    await expect(tv.getByTestId('ec-storyline')).toBeVisible()
    const [th, sl] = await page.evaluate(() => {
      const v = document.querySelector('[data-testid="topic-view"]')!
      const r = (id: string) => v.querySelector(`[data-testid="${id}"]`)!.getBoundingClientRect()
      const [a, b] = [r('ec-theme'), r('ec-storyline')]
      return [{ y: a.y, height: a.height }, { y: b.y }]
    })
    expect(sl.y, 'storyline under theme').toBeGreaterThanOrEqual(th.y + th.height)
    await pair(`/theme/${encodeURIComponent('tc:safety-practices')}`, 'theme-view', 'theme-topics', 'ec-top-voices')
    await pair(`/storyline/${encodeURIComponent('topic:risk-management')}`, 'storyline-view', 'storyline-topics', 'ec-top-voices')
  })
}

/**
 * One page width (operator 2026-10-05): every page fills the app shell, so on desktop every page's
 * content starts at the SAME left edge — the shell's. Measured on each page's first child of <main>.
 */
test('every full-width page starts at the shell edge on desktop', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await signInIsolated(page, 'one-page-width', testInfo)
  const lefts: Record<string, number> = {}
  for (const url of [
    '/',
    '/browse',
    '/catalog',
    '/search?q=reliability',
    '/queue',
    '/library',
    '/podcast/p05',
    `/topic/${encodeURIComponent('topic:risk-management')}`,
    `/person/${encodeURIComponent('person:nora')}`,
    `/theme/${encodeURIComponent('tc:safety-practices')}`,
    `/storyline/${encodeURIComponent('topic:risk-management')}`,
  ]) {
    await page.goto(url)
    await page.locator('main > *').first().waitFor()
    lefts[url] = await page.evaluate(() => Math.round(document.querySelector('main > *')!.getBoundingClientRect().left))
  }
  const edge = lefts['/']
  for (const [url, left] of Object.entries(lefts)) expect(left, `${url} left edge`).toBe(edge)
})

test('long reading text keeps a readable width; sign-in fills a phone and centres its contents', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/about/privacy')
  const prose = await page.getByTestId('privacy-policy').evaluate((e) => e.getBoundingClientRect().width)
  expect(prose, 'Privacy text column').toBeLessThanOrEqual(672)
  await page.goto('/login')
  const view = page.getByTestId('login-view')
  await expect(view.getByRole('heading', { level: 1 })).toHaveCSS('text-align', 'center')
  await expect(page.getByTestId('login-legal')).toHaveCSS('text-align', 'center')
  await page.setViewportSize({ width: 412, height: 915 })
  await page.goto('/login')
  const b = (await view.boundingBox())!
  expect(Math.round(b.x)).toBeLessThanOrEqual(20)
  expect(Math.round(412 - (b.x + b.width))).toBeLessThanOrEqual(20)
})

/**
 * Offline ("On this device") is a centred 448px column with its title and messages centred, like
 * sign-in. It renders only offline AND signed out, so: load it once online (its chunk loads, then it
 * redirects away), cut the network, and route to it in-app.
 */
test('the offline page is a centred column with centred text', async ({ page, context }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await page.goto('/offline')
  await page.waitForURL((u) => !u.pathname.startsWith('/offline'))
  await context.setOffline(true)
  try {
    await page.evaluate(() => {
      history.pushState({}, '', '/offline')
      window.dispatchEvent(new PopStateEvent('popstate', { state: history.state }))
    })
    const view = page.getByTestId('offline-downloads')
    await expect(view).toBeVisible()
    const b = (await view.boundingBox())!
    expect(b.width).toBeLessThanOrEqual(448)
    expect(Math.abs(b.x - (1440 - (b.x + b.width))), 'centred').toBeLessThanOrEqual(2)
    await expect(view.getByRole('heading', { level: 1 })).toHaveCSS('text-align', 'center')
  } finally {
    await context.setOffline(false)
  }
})
