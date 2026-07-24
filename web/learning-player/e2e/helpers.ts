import { expect, type Page, type TestInfo } from '@playwright/test'

/**
 * Sign in as an ISOLATED mock identity, unique per (spec, project). The mock OAuth provider honours
 * the `?as=` hint (dev/e2e only) and self-completes as `e2e-<hint>` — so parallel specs don't share
 * one mock user (which would race on the shared per-user files). `who` should be the spec's name.
 */
export async function signInIsolated(page: Page, who: string, testInfo: TestInfo): Promise<void> {
  const id = `${who}-${testInfo.project.name}`.toLowerCase().replace(/[^a-z0-9-]/g, '')
  await page.goto(`/api/app/auth/login?as=${encodeURIComponent(id)}`)
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible()
}

/**
 * Reveal the transcript when it's collapsed.
 *
 * The transcript is opt-in on mobile (a "Show transcript" toggle) so pressing
 * play doesn't jump the listener into it; on desktop it's the always-visible
 * side column and the toggle is hidden. This clicks the toggle only when it's
 * actually visible — a no-op on desktop — so transcript specs pass under both
 * Playwright projects (mobile-chrome + desktop-chrome).
 */
export async function openTranscript(page: Page): Promise<void> {
  const toggle = page.getByTestId('transcript-toggle')
  // Wait for the toggle to attach — it renders once segments load, on BOTH viewports
  // (lg:hidden on desktop). Then click only if it's actually visible (mobile); on desktop
  // it's attached-but-hidden and the transcript is already the side column, so this no-ops.
  await toggle.waitFor({ state: 'attached', timeout: 15_000 }).catch(() => {})
  if (await toggle.isVisible().catch(() => false)) {
    await toggle.click()
  }
}

/**
 * Route the episode audio-source to a valid silent WAV.
 *
 * The committed fixture ships a data-URL MP3 that headless Chromium can't decode,
 * so the player flips to `audioError` and renders the error message instead of the
 * transport panel. On a real device the bridged audio decodes and the panel (which
 * now hosts the transcript toggle) renders. Routing a decodable WAV makes the e2e
 * exercise that real, audio-present player. Only the external media bytes are
 * substituted — the corpus/API paths (segments, metadata, insights) stay real.
 *
 * Call BEFORE the first navigation so the route is in place when the player loads.
 */
export async function routeLoadableAudio(page: Page, seconds = 60, rate = 8000): Promise<void> {
  const dataSize = seconds * rate // 8-bit mono
  const buf = Buffer.alloc(44 + dataSize)
  buf.write('RIFF', 0)
  buf.writeUInt32LE(36 + dataSize, 4)
  buf.write('WAVE', 8)
  buf.write('fmt ', 12)
  buf.writeUInt32LE(16, 16)
  buf.writeUInt16LE(1, 20) // PCM
  buf.writeUInt16LE(1, 22) // mono
  buf.writeUInt32LE(rate, 24)
  buf.writeUInt32LE(rate, 28)
  buf.writeUInt16LE(1, 32)
  buf.writeUInt16LE(8, 34)
  buf.write('data', 36)
  buf.writeUInt32LE(dataSize, 40)
  buf.fill(128, 44) // 8-bit silence
  const url = `data:audio/wav;base64,${buf.toString('base64')}`
  await page.route('**/audio-source', (route) =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ url }) }),
  )
}
