import { expect, test } from '@playwright/test'

/**
 * The player loads the corpus's real audio (#1618).
 *
 * The fixture used to carry a 146-byte `data:audio/mpeg` URI — an ID3 header with no audio frames,
 * which no browser can decode — so the player fell into its error state and the suite compensated
 * by route-fulfilling a synthetic WAV over the audio-source response. Nine specs depended on that
 * stub, in a suite whose contract is that it has no mocks.
 *
 * That made the workaround self-concealing: every transport assertion was really exercising the
 * stub, so a corpus whose audio does not work looked exactly like one whose audio does. Worse, it
 * hid that 46 real MP3s covering every episode had been committed all along, one directory up from
 * where anyone looked.
 *
 * `content.media_url` is now a relative `/audio/<episode_id>.mp3` — the same convention the RSS
 * fixtures use for enclosures, so no host or port is baked into 36 committed files — and the app's
 * `/audio` proxy forwards it to the mock podcast host.
 *
 * This asserts the chain end to end, by name, so a regression fails here rather than quietly
 * reinstating the error panel everywhere.
 */
test('the corpus audio is real, served, and playable', async ({ page }) => {
  await page.goto('/podcast/p05')
  await page.getByText('Index Investing Without the Myths').first().click()
  await expect(page).toHaveURL(/\/episode\//)

  // 1. The API hands back a relative path, not a data URI and not a hard-coded host.
  const slug = new URL(page.url()).pathname.split('/').pop() as string
  const res = await page.request.get(`/api/app/episodes/${slug}/audio-source`)
  expect(res.ok(), 'audio-source must resolve').toBe(true)
  const { url } = (await res.json()) as { url: string }
  expect(url, 'media_url must be a relative /audio/ path').toMatch(/^\/audio\/[\w-]+\.mp3$/)

  // 2. The proxy actually serves bytes, and they are audio.
  const media = await page.request.get(url)
  expect(media.status(), `${url} must be served by the mock podcast host`).toBe(200)
  expect(media.headers()['content-type']).toContain('audio')
  expect(Number(media.headers()['content-length'] ?? 0)).toBeGreaterThan(50_000)

  // 3. A real decoder accepts it — the property the placeholder failed, and the only one that
  //    matters to a listener.
  const verdict = await page.evaluate(async (src) => {
    const a = new Audio()
    a.src = src
    return await new Promise<{ outcome: string; duration: number | null }>((resolve) => {
      const t = setTimeout(() => resolve({ outcome: 'timeout', duration: null }), 15_000)
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
  // full-listen.spec seeks to 30s; anything shorter makes that assertion meaningless.
  expect(verdict.duration ?? 0).toBeGreaterThan(31)

  // 4. And the player renders its transport rather than the audio-error state.
  await expect(page.getByRole('button', { name: 'Play', exact: true }).first()).toBeVisible()
})
