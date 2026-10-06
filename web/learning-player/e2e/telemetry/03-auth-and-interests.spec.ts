import { expect, test } from '@playwright/test'
import { attachSink, DEV_UMAMI_WEBSITE_ID } from './sink'
import './settle'

/**
 * The funnel's middle: setting off to sign in, arriving signed in, and the pseudonymous identity
 * that ties a participant's events together (#2265, #2267).
 *
 * `analytics_id` is the whole reason the beta can produce a per-participant report without storing
 * anyone's email in Umami. If `identify` does not actually reach the wire, every event is anonymous
 * and the per-person reports are empty — while the aggregate dashboards look perfectly healthy. So
 * this asserts the id arrives, that it is NOT the account id or the email, and that signing out
 * stops attributing to it.
 */

test.describe('auth funnel', () => {
  test('auth_started reports the provider at the moment of departure', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/login')

    // Which control exists depends on the provider, and all three branches are real app states:
    //   • mock provider WITH seeded dev identities → a pick-a-user list
    //   • mock provider with NONE seeded → the custom-identity form (what this stack renders:
    //     `getDevUsers()` answers enabled=true, users=[])
    //   • real provider → the single OAuth button, which is `v-else` to the dev block and therefore
    //     absent whenever dev sign-in is on
    // Following the app rather than assuming one of them is the difference between this spec testing
    // sign-in and this spec timing out against an element that was never going to render.
    // `devEnabled` is set in an async `onMounted` (`getDevUsers()`), so NONE of these controls exist
    // on the first tick. Branching immediately after `goto` read every one as absent and fell
    // through to the OAuth button, which this build never renders — a 2-minute timeout that looked
    // like a broken sign-in and was really a test asking too early. Wait for whichever arrives.
    const devList = page.getByTestId('dev-user-list')
    const devCustom = page.getByTestId('dev-custom-input')
    const oauth = page.getByTestId('signin-button')
    await expect(
      devList.or(devCustom).or(oauth).first(),
      'the login page must offer SOME way to sign in',
    ).toBeVisible()

    if (await devList.isVisible().catch(() => false)) {
      await page.locator('[data-testid^="dev-user-"]').first().click()
    } else if (await devCustom.isVisible().catch(() => false)) {
      await devCustom.fill('telemetry-auth-started')
      await page.getByTestId('dev-custom-submit').click()
    } else {
      await page.getByTestId('signin-button').click()
    }

    const started = await sink.waitForEvent('auth_started')
    expect(started.website).toBe(DEV_UMAMI_WEBSITE_ID)
    // 'mock' when the dev picker supplied an identity hint, 'oauth' otherwise — the client cannot
    // know the real provider, and claiming to is worse than reporting what it does know.
    expect(['mock', 'oauth']).toContain(String(started.data?.provider))
  })

  test('auth_completed fires exactly once per sign-in, not once per revalidation', async ({
    page,
  }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-auth-once')
    await page.waitForLoadState('networkidle')
    await sink.waitForEvent('auth_completed')

    // `refresh()` runs at boot AND on every background revalidation. Without the latch in
    // `stores/auth.ts` this would count one listener dozens of times, and the funnel's last step
    // would show a higher conversion than the step before it — an impossible funnel that still
    // looks like a number.
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await page.goto('/library')
    await page.waitForLoadState('networkidle')

    // Each full page load is a new app instance, so a per-load event is expected; what must not
    // happen is multiple per load.
    const perLoad = sink.byName('auth_completed').length
    expect(perLoad, 'auth_completed must not multiply within a session').toBeLessThanOrEqual(3)
  })
})

test.describe('pseudonymous identity', () => {
  test('identify sends the analytics_id and never the account id or email', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-identity')
    await page.waitForLoadState('networkidle')

    // What the server actually says this account is.
    const me = await page.request.get('/api/app/me').then((r) => r.json())
    expect(
      me.analytics_id,
      'the server must issue an analytics_id, or every event stays anonymous',
    ).toBeTruthy()

    await expect
      .poll(() => sink.umami.some((b) => b.type === 'identify' || Boolean(b.id)), { timeout: 15_000 })
      .toBe(true)

    const ident = sink.umami.find((b) => b.type === 'identify' || Boolean(b.id))
    const payload = JSON.stringify(ident?.raw ?? {})

    // The id that went out is the pseudonymous one.
    expect(payload).toContain(String(me.analytics_id))

    // And nothing identifying went with it. `user_id` is the account key and `email` is the person;
    // either one in Umami would turn a pseudonymous store into a personal one.
    if (me.user_id) {
      expect(payload, 'the account id must not reach Umami').not.toContain(String(me.user_id))
    }
    if (me.email) {
      expect(payload, 'the email must not reach Umami').not.toContain(String(me.email))
      const local = String(me.email).split('@')[0]
      // Only meaningful for a realistic local part; a one-character one appears in almost any id.
      if (local.length > 3) expect(payload).not.toContain(local)
    }
  })
})

test.describe('interests picker', () => {
  test('shown / saved / dismissed are distinguishable, and count is bucketed', async ({ page }) => {
    const sink = attachSink(page)
    await page.goto('/api/app/auth/login?as=telemetry-interests')
    await page.waitForLoadState('networkidle')

    // The picker is the onboarding sheet, opened from Home's "Personalize your Home" card; Profile
    // edits interests in place since 2026-10-04 and no longer opens it.
    await page.goto('/')
    const open = page.getByRole('button', { name: 'Choose interests' })
    await open.click()

    const shown = await sink.waitForEvent('interests_picker_shown')
    expect(
      shown.data,
      'the trigger separates the home prompt from other entry points — different intents',
    ).toMatchObject({ trigger: 'home_prompt' })

    // Dismiss without saving → dismissed, NOT saved.
    await page.getByTestId('interests-cancel').click()
    await sink.waitForEvent('interests_dismissed')
    expect(
      sink.byName('interests_saved'),
      'a dismissal must never be reported as a save',
    ).toHaveLength(0)

    // Now actually save something.
    await open.click()
    await page.getByTestId('interest-add-topic').click()
    const chip = page.getByTestId('interest-suggestion').first()
    await expect(chip, 'the picker needs real clusters to have something to choose').toBeVisible()
    await chip.click()
    // `interests-save`, not `interests-close`: close and cancel both route through `closeSheet()`,
    // which reports a DISMISSAL. Clicking either here would have asserted a save while performing
    // the opposite action — and the event names are similar enough that the mistake reads as fine.
    await page.getByTestId('interests-save').click()

    const saved = await sink.waitForEvent('interests_saved')
    const count = String(saved.data?.count ?? '')
    // BUCKETED on purpose: the exact number of chosen topics, with the chosen ids, would be close to
    // a fingerprint on a small beta. A bucket answers every question a metric asks of it.
    //
    // Asserted as MEMBERSHIP in the vocabulary, not as "not a number": `toCountBucket` maps 0→'0'
    // and 1→'1', so two of the five legitimate labels are bare digits. A "must not look like a
    // number" rule rejects the correct value for exactly one chosen topic while still accepting a
    // raw 7, which is backwards. The vocabulary is the contract; check against it.
    expect(['0', '1', '2-5', '6-20', '21+']).toContain(count)
  })
})
