/**
 * Topic vs theme vs storyline — the three pages side by side, for a design decision.
 *
 * These three are the hardest thing in the product to tell apart, and the UI currently makes it
 * harder rather than easier:
 *
 *   TOPIC      one subject. An entity: it has a page, a follow, episodes of its own.
 *   THEME      topics that MEAN the same thing (cosine similarity). A grouping, not an entity.
 *   STORYLINE  topics that KEEP COMING UP TOGETHER (co-occurrence). Also a grouping.
 *
 * Storyline already has its own route and view. **Theme has neither** — a `tc:` id falls through
 * to `/topic/:id` and renders in the entity card, which is why its header reads "TOPIC" and why it
 * looks like a topic that happens to list a lot of similar topics.
 *
 * The fixture is chosen so all three shots describe the SAME subject from three angles:
 * `topic:risk-management` is a topic, a member of the `tc:show-themes` theme, and the anchor of the
 * `thc:managing-risk` storyline. Any difference between the three images is a difference in how the
 * product presents the three ideas, not in the underlying data — which is the only way to judge
 * whether they are distinguishable.
 *
 * Not an assertion suite. It produces images; a person decides.
 */
import { expect, test, type Page } from '@playwright/test'

const VARIANT = process.env.DESIGN_VARIANT || process.env.DESIGN_DIRECTION || 'baseline'
const dir = (name: string) =>
  `design-results/${VARIANT}/${test.info().project.name}/${name}.png`

/** From `tests/fixtures/app-validation-corpus/v3` — read out of the artifacts, not invented. */
const TOPIC = 'topic:risk-management'
const THEME = 'tc:show-themes' //      "Show Themes", 5 members
const STORYLINE_ANCHOR = TOPIC //      /storyline/:id takes the ANCHOR TOPIC, not the thc: id

const IDENTITY = 'design'

async function signIn(page: Page): Promise<void> {
  await page.goto(`/api/app/auth/login?as=${IDENTITY}`)
  await page.goto('/')
}

/** Let fonts, images and any entry animation finish before the shutter. */
async function settle(page: Page): Promise<void> {
  await page.waitForLoadState('networkidle').catch(() => undefined)
  await page.evaluate(() => document.fonts?.ready).catch(() => undefined)
  await page.waitForTimeout(400)
}

async function shoot(page: Page, name: string): Promise<void> {
  await settle(page)
  await page.screenshot({ path: dir(`${name}-full`), fullPage: true })
  await page.screenshot({ path: dir(`${name}-viewport`), fullPage: false })
}

test('cluster-topic', async ({ page }) => {
  await signIn(page)
  await page.goto(`/topic/${encodeURIComponent(TOPIC)}`)
  // Proof the page loaded its subject rather than an error card — otherwise the shot is of a
  // failure state and the comparison is worthless.
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await shoot(page, 'cluster-1-topic')
})

test('cluster-theme', async ({ page }) => {
  await signIn(page)
  // NOTE the route: a theme has no page of its own, so it borrows the topic one. That is the
  // defect this shot is here to show, not a shortcut taken by the test.
  await page.goto(`/topic/${encodeURIComponent(THEME)}`)
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await shoot(page, 'cluster-2-theme')
})

test('cluster-storyline', async ({ page }) => {
  await signIn(page)
  await page.goto(`/storyline/${encodeURIComponent(STORYLINE_ANCHOR)}`)
  await shoot(page, 'cluster-3-storyline')
})
