import { expect, test } from '@playwright/test'

test.describe('Stack smoke test', () => {
  test('Nginx serves SPA shell', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('body')).toBeVisible()
  })

  test('API health via Nginx proxy', async ({ request }) => {
    const res = await request.get('/api/health')
    expect(res.ok()).toBeTruthy()
    const body = await res.json()
    expect(body).toHaveProperty('status')
  })

  test('graph canvas is present after load', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('.graph-canvas')).toBeVisible({ timeout: 60_000 })
  })

  test('corpus run summary is visible to API', async ({ request }) => {
    const res = await request.get('/api/corpus/documents/run-summary')
    const text = await res.text()
    if (res.status() !== 200) {
      throw new Error(`run-summary ${res.status()}: ${text.slice(0, 500)}`)
    }
    const body = JSON.parse(text) as unknown
    expect(body).toBeTruthy()
  })
})
