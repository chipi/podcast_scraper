import { expect, test } from '@playwright/test'
import { expectSignedIn, signInIsolated } from './helpers'

/**
 * Copy link / Copy text, and what happens to the person the link is sent to (operator 2026-10-05).
 *
 * Beta testers could not tell "Share link" and "Share text" from "Share card": on a phone all three
 * opened the same OS sheet. They COPY now. The link is the one https URL for the thing — the same
 * link the emails carry and the installed app claims — so this walks the web half of the flow end
 * to end: copy it, open it signed out, sign in, land on the same episode. (The in-app half needs the
 * deployed /.well-known files and a device; see docs/wip/APP-LINKS-PRE-BUILD-TODO-2026-10-05.md.)
 */
test('Copy link and Copy text put the episode on the clipboard, and the link opens it — through sign-in', async ({
  page,
  context,
  browser,
}, testInfo) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await signInIsolated(page, 'copy-link', testInfo)
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)
  const path = new URL(page.url()).pathname

  await page.getByTestId('share-menu').click()
  await expect(page.getByTestId('share-link')).toHaveText('Copy link')
  await page.getByTestId('share-link').click()
  await expect(page.getByRole('status').filter({ hasText: 'Link copied' })).toBeVisible()
  const link = await page.evaluate(() => navigator.clipboard.readText())
  expect(link).toBe(new URL(path, page.url()).href)

  await page.getByTestId('share-menu').click()
  await page.getByTestId('share-text').click()
  await expect(page.getByRole('status').filter({ hasText: 'Text copied' })).toBeVisible()
  const text = await page.evaluate(() => navigator.clipboard.readText())
  // "Episode title — Show", then the link: what someone pastes into a message.
  expect(text.split('\n')).toEqual([expect.stringMatching(/^Index Investing Without the Myths — .+/), link])

  // The recipient, signed out: the sign-in gate, carrying the link through.
  const other = await browser.newContext({ baseURL: new URL(link).origin })
  const them = await other.newPage()
  await them.goto(link)
  await expect(them).toHaveURL(/\/welcome/)
  const redirect = new URL(them.url()).searchParams.get('redirect')
  expect(redirect).toBe(path)
  const id = `copy-link-recipient-${testInfo.project.name}`.toLowerCase().replace(/[^a-z0-9-]/g, '')
  await them.goto(`/api/app/auth/login?as=${encodeURIComponent(id)}&return_to=${encodeURIComponent(redirect!)}`)
  await expect(them).toHaveURL(new RegExp(`${path}$`))
  await expectSignedIn(them)
  await expect(them.getByText('Index Investing Without the Myths').first()).toBeVisible()
  await other.close()
})

test('a topic link with the full graph id opens the topic — the form the emails now send', async ({
  page,
}, testInfo) => {
  // The emails sent /topic/<bare-slug>, which the topic page cannot resolve: an empty page.
  await signInIsolated(page, 'copy-link-topic', testInfo)
  await page.goto(`/topic/${encodeURIComponent('topic:reliability')}`)
  await expect(page.getByTestId('topic-view')).toBeVisible()
  await expect(page.getByTestId('topic-view').getByText('reliability', { exact: false }).first()).toBeVisible()
})
