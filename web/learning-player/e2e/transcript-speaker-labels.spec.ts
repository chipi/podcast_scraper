import { expect, test } from '@playwright/test'
import { openTranscript, signInIsolated } from './helpers'

/**
 * A raw diarization label is never shown as a speaker (RFC-131 S3, operator decision 2026-10-30).
 *
 * Fixture p10 (a Spanish show) has two voices the roster could not name: segment labels
 * `SPEAKER_00` / `SPEAKER_01` with no voice type. The transcript used to print those ids. It now
 * reads "Unidentified speaker 1" / "Unidentified speaker 2", numbered by first appearance, and the
 * label comes from the client's i18n (speaker_kind / speaker_index), not the server's English.
 */
test('unnamed voices read as numbered unidentified speakers, never raw labels', async ({
  page,
}, testInfo) => {
  await signInIsolated(page, 'speaker-labels', testInfo)
  await page.goto('/')
  await page.goto('/podcast/p10')
  await page.getByText('Construyendo Senderos Que Duran').first().click()
  await openTranscript(page)
  const transcript = page.getByTestId('transcript')
  await expect(transcript.locator('.lp-speaker').first()).toBeVisible()

  const speakers = await transcript.locator('.lp-speaker').allInnerTexts()
  expect(speakers.length).toBeGreaterThan(0)
  for (const s of speakers) {
    expect(s, `raw label shown as a speaker: ${s}`).not.toMatch(/^(speaker[_\s-]*\d+|unknown_guest_\d+)$/i)
  }
  expect(speakers.some((s) => /^Unidentified speaker [12]$/.test(s))).toBe(true)
  await expect(transcript).not.toContainText(/SPEAKER_\d+/)
  const firstUnnamed = transcript.locator('.lp-speaker', { hasText: /^Unidentified speaker \d+$/ }).first()
  await firstUnnamed.scrollIntoViewIfNeeded()
  await page.screenshot({ path: testInfo.outputPath('transcript-speakers.png') })
})
