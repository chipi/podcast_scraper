import { expect, test } from '@playwright/test'

/**
 * The corpus's own audio is playable (#1618).
 *
 * The fixture shipped a 146-byte `data:audio/mpeg` URI — an ID3 header with no audio frames — so no
 * browser could decode it. The player flipped to its error state and rendered the error panel
 * instead of the transport, and the suite compensated by route-fulfilling a synthetic WAV over the
 * audio-source response. Eight specs depended on that stub, in a suite whose contract is that it
 * has no mocks: it bootstraps a real backend from real fixtures.
 *
 * That made the workaround self-concealing. Every spec that touched the transport was really
 * testing the stub, so a corpus whose audio does not work looked exactly like one whose audio does.
 * This asserts the property directly, so a regression fails here by name rather than quietly
 * reintroducing the error panel everywhere.
 */
test('the episode audio the API serves actually decodes and plays', async ({ page }) => {
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  // Straight from the API — no interception anywhere in this file.
  const slug = new URL(page.url()).pathname.split('/').pop() as string
  const res = await page.request.get(`/api/app/episodes/${slug}/audio-source`)
  expect(res.ok(), 'audio-source must resolve').toBe(true)
  const { url } = (await res.json()) as { url: string }
  expect(url, 'the fixture must carry an mpeg data URI').toContain('data:audio/mpeg')

  // Decodability is the whole point, and only a real decoder can answer it.
  const verdict = await page.evaluate(async (src) => {
    const a = new Audio()
    a.src = src
    return await new Promise<{ outcome: string; duration: number | null }>((resolve) => {
      const t = setTimeout(() => resolve({ outcome: 'timeout', duration: null }), 10_000)
      a.addEventListener('loadedmetadata', () => {
        clearTimeout(t)
        resolve({ outcome: 'ok', duration: a.duration })
      })
      a.addEventListener('error', () => {
        clearTimeout(t)
        resolve({ outcome: `error:${a.error?.code ?? '?'}`, duration: null })
      })
      a.load()
    })
  }, url)

  expect(verdict.outcome, 'the corpus audio must decode in a real browser').toBe('ok')
  // full-listen.spec seeks to 30s, so the fixture has to be longer than that to mean anything.
  expect(verdict.duration ?? 0).toBeGreaterThan(31)

  // And the player renders its transport rather than the audio-error state.
  await expect(page.getByRole('button', { name: 'Play', exact: true }).first()).toBeVisible()
})
