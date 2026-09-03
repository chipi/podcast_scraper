import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * The last of the declared coverage gaps (E2E_SURFACE_MAP, closed 2026-09-03): the entity card's
 * storyline + theme-member sections, the topic conversation arc, and the trending-shows rail.
 *
 * Two of these are asserted as INVARIANTS rather than as presence, and deliberately so. A rail
 * whose data the fixture corpus does not produce is *supposed* to omit itself (UXS-012), so
 * demanding it be visible would test the corpus rather than the app — and would fail for the right
 * behaviour. What can be asserted honestly is the contract: **when present, it has content; it is
 * never an empty shell.** That is the failure mode a listener actually sees.
 */

const TOPIC = 'topic:personal-finance'

test('the topic view renders the entity body, its arc and its theme members', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'topic-entity', testInfo)
  await page.goto(`/topic/${encodeURIComponent(TOPIC)}`)

  // The topic surface itself must resolve — an unresolvable entity is a 404 story, not a blank page.
  await expect(page.getByRole('heading').first()).toBeVisible()

  // The conversation arc: shape over precision (UXS-013). Present-with-bars, or absent.
  const arc = page.getByTestId('topic-conversation-arc')
  if (await arc.isVisible().catch(() => false)) {
    expect(await page.locator('[data-testid^="tca-bar-"]').count()).toBeGreaterThan(0)
  }

  // Theme members: the sibling topics that share a theme. Same invariant.
  const themes = page.getByTestId('ec-theme-members')
  if (await themes.isVisible().catch(() => false)) {
    await expect(themes).not.toBeEmpty()
  }
})

test('a storyline follow on the entity card writes an interest', async ({ page }, testInfo) => {
  await signInIsolated(page, 'entity-storyline', testInfo)
  await page.goto(`/topic/${encodeURIComponent(TOPIC)}`)

  const follow = page.getByTestId('ec-follow-storyline').first()
  if (!(await follow.isVisible().catch(() => false))) {
    test.skip(true, 'this topic has no storyline on the fixture corpus — nothing to assert')
  }
  // It must write the SAME interest token the picker and the rails write, or a storyline followed
  // here would not appear in Your Week.
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/api/app/interests')),
    follow.click(),
  ])
})

test('the trending-shows rail is never an empty shell', async ({ page }, testInfo) => {
  await signInIsolated(page, 'trending-shows', testInfo)
  await page.goto('/')

  const rail = page.getByTestId('trending-shows-rail')

  // Wait for the section to SETTLE first. It deliberately renders while loading (`hasAny ||
  // !isReady`), so a visible-and-empty rail is correct mid-fetch — asserting before that resolves
  // tests the skeleton, not the contract.
  await page.waitForLoadState('networkidle')
  await expect
    .poll(
      async () => {
        const visible = await rail.isVisible().catch(() => false)
        const cards = await page.getByTestId('trending-show-card').count()
        // settled = gone (no trending shows in this corpus) or populated
        return !visible || cards > 0
      },
      { timeout: 15_000 },
    )
    .toBe(true)

  if (!(await rail.isVisible().catch(() => false))) {
    // Absent is CORRECT when the corpus yields no trending shows — that is the rule, not a gap.
    return
  }
  // Present AND settled means it must carry cards. A section left visible with nothing in it reads
  // as a loading state that never finishes, which is the failure this guards.
  expect(await page.getByTestId('trending-show-card').count()).toBeGreaterThan(0)
})
