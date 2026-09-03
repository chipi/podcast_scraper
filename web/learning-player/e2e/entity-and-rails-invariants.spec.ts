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

// A topic the fixture corpus gives BOTH a conversation arc and theme siblings. Chosen by
// probing: `topic:personal-finance` legitimately has neither, so a spec pointed at it could only
// ever assert conditionally — i.e. pass whether or not the sections ever render again. Naming a
// topic that has them is what makes the assertions below unconditional and therefore real.
const TOPIC = 'topic:risk-management'

test('the topic view renders the entity body, its arc and its theme members', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'topic-entity', testInfo)
  await page.goto(`/topic/${encodeURIComponent(TOPIC)}`)

  // The topic surface itself must resolve — an unresolvable entity is a 404 story, not a blank page.
  await expect(page.getByRole('heading').first()).toBeVisible()
  // SETTLE before branching. Reading `isVisible()` straight after `goto` resolves false mid-load,
  // so both conditionals below would skip and a regression where these sections never render
  // again would pass silently (advisor-2 #7). The whole value of the test is in the branches.
  await page.waitForLoadState('networkidle')

  // The conversation arc: shape over precision (UXS-013). Present-with-bars, or absent — and the
  // corpus decides which, so at least ONE of the two knowledge sections must be present or this
  // is not a topic page worth asserting against.
  const arc = page.getByTestId('topic-conversation-arc')
  const themes = page.getByTestId('ec-theme-members')
  // UNCONDITIONAL, because this topic has both. If either stops rendering, this fails — which is
  // the whole point; a conditional here would pass through the regression it exists to catch.
  await expect(arc).toBeVisible()
  expect(await page.locator('[data-testid^="tca-bar-"]').count()).toBeGreaterThan(0)
  await expect(themes).toBeVisible()
  await expect(themes).not.toBeEmpty()
})

test('a storyline follow on the entity card writes an interest', async ({ page }, testInfo) => {
  await signInIsolated(page, 'entity-storyline', testInfo)
  await page.goto(`/topic/${encodeURIComponent(TOPIC)}`)

  // Asserted, not guarded. The button renders on `auth.isAuthenticated && themeClusterId`, and
  // the fixture corpus DOES carry a theme cluster for this topic: enrichments/
  // topic_theme_clusters.json defines `thc:managing-risk` with topic:risk-management among its
  // three members. The spec signs in, so both halves hold and the button must be there. The
  // previous `if (!visible) skip` could only ever hide a regression.
  const follow = page.getByTestId('ec-follow-storyline').first()
  await expect(follow).toBeVisible()
  // It must write the SAME interest token the picker and the rails write, or a storyline followed
  // here would not appear in Your Week. Asserted as a SUCCESSFUL WRITE: waiting for any response
  // whose URL contains `/api/app/interests` was satisfied by an unrelated GET, or by a 4xx — so
  // the invariant in this comment was not actually being checked (advisor-2 #7).
  const [response] = await Promise.all([
    page.waitForResponse(
      (r) =>
        r.url().includes('/api/app/interests') && r.request().method() !== 'GET' && r.ok(),
    ),
    follow.click(),
  ])
  expect(response.ok()).toBe(true)
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
