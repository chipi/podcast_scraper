import { expect, test } from '@playwright/test'
import { signInIsolated } from './helpers'

/**
 * Delete account, end to end against the real server (#2273, App Store 5.1.1(v)).
 *
 * The unit tests prove the screen and the server's purge separately; only this proves the person
 * can find it inside the app, that the typed confirmation gates it, and that afterwards the session
 * is really gone — the requirement is about the in-app path, not the endpoint.
 */
test('a signed-in person deletes their account from Profile and is signed out', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'delete-account', testInfo)
  await page.goto('/profile?tab=account')
  await page.getByTestId('profile-delete-account').click()
  await expect(page).toHaveURL(/\/account\/delete$/)

  const submit = page.getByTestId('delete-account-submit')
  await expect(page.getByTestId('delete-account-who')).toContainText('This deletes that account.')
  await expect(submit).toBeDisabled()
  await page.getByTestId('delete-account-confirm').fill('delete')
  await expect(submit).toBeDisabled()
  await page.getByTestId('delete-account-confirm').fill('DELETE')
  await expect(submit).toBeEnabled()
  await submit.click()

  await expect(page).toHaveURL(/\/welcome\?deleted=1$/)
  await expect(page.getByTestId('landing-account-deleted')).toHaveText('Your account was deleted.')
  // The server agrees: the session no longer resolves to anyone.
  const me = await page.request.get('/api/app/me')
  expect(me.status()).toBe(401)
})

test('the deletion page explains itself to someone signed out (the Play listing link)', async ({
  page,
}) => {
  await page.goto('/account/delete')
  await expect(page.getByTestId('delete-account-signed-out')).toContainText('Profile › Account › Delete account')
  await expect(page.getByTestId('delete-account-submit')).toHaveCount(0)
})

test('clear listening history asks first, then clears, and the account stays', async ({ page }, testInfo) => {
  await signInIsolated(page, 'clear-history', testInfo)
  await page.goto('/profile?tab=account')
  await page.getByTestId('profile-clear-history-open').click()
  await expect(page.getByTestId('profile-clear-history')).toContainText('library, queue')
  await page.getByTestId('profile-clear-history-confirm').click()
  await expect(page.getByTestId('profile-clear-history-result')).toHaveText(
    'Your listening history was cleared.',
  )
  const me = await page.request.get('/api/app/me')
  expect(me.status()).toBe(200)
})

test('the privacy policy is readable signed out, at /privacy (the store-listing URL)', async ({ page }) => {
  await page.goto('/privacy')
  await expect(page).toHaveURL(/\/about\/privacy$/)
  await expect(page.getByTestId('privacy-policy')).toContainText('Delete your account')
  await expect(page.getByTestId('privacy-draft-notice')).toBeVisible()
})
