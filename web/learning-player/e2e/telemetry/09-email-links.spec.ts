import { expect, test } from '@playwright/test'
import { attachSink } from './sink'
import './settle'

/**
 * Which links people click in the emails we send (operator 2026-10-05).
 *
 * The delivery renderer tags every content link with utm_source=email + which email + what kind of
 * page; the app reports `email_link_opened` when the link lands. Against the REAL dev Umami: a click
 * from a signed-out reader is counted at the sign-in gate, and again — signed in — when sign-in
 * sends them on to the page. That pair is the email → sign-in funnel.
 */
const LINK =
  '/episode/p05?utm_source=email&utm_campaign=your_week_digest&utm_content=episode'

test('an email click is counted signed out at the gate, then signed in on arrival', async ({ page }) => {
  const sink = attachSink(page)
  await page.goto(LINK)
  await expect(page).toHaveURL(/\/welcome/)
  const out = await sink.waitForEvent('email_link_opened', 20_000)
  expect(out.data).toEqual({
    campaign: 'your_week_digest',
    element: 'episode',
    target: 'player',
    signed_in: false,
  })

  const redirect = new URL(page.url()).searchParams.get('redirect')!
  await page.goto(`/api/app/auth/login?as=telemetry-email-link&return_to=${encodeURIComponent(redirect)}`)
  await expect(page).toHaveURL(/\/episode\//)
  await expect
    .poll(() => sink.byName('email_link_opened').length, { timeout: 20_000 })
    .toBe(2)
  expect(sink.byName('email_link_opened')[1].data).toEqual({
    campaign: 'your_week_digest',
    element: 'episode',
    target: 'player',
    signed_in: true,
  })
})
