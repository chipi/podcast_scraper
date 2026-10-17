import { beforeEach, describe, expect, it, vi } from 'vitest'

// PLAYER-1I (2026-10-17): resuming playback while the app was in the background made Android
// refuse the foreground service, and the refusal crashed the app. The native side now answers
// `foreground: false` instead; the player reports that once per launch and carries on.

const start = vi.fn()
vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: () => true, getPlatform: () => 'android' },
  CapacitorCookies: { setCookie: vi.fn(async () => {}) },
  registerPlugin: () => ({ start: (...a: unknown[]) => start(...a) }),
}))
const captureMessage = vi.fn()
vi.mock('@sentry/capacitor', () => ({ captureMessage: (...a: unknown[]) => captureMessage(...a) }))
vi.mock('./api', () => ({ setAuthToken: vi.fn() }))

beforeEach(() => {
  vi.resetModules()
  start.mockReset()
  captureMessage.mockReset()
})

describe('startBackgroundAudio on Android', () => {
  it('reports a refused foreground once per launch, and does not throw', async () => {
    const { startBackgroundAudio } = await import('./native')
    start.mockResolvedValue({ foreground: false })
    await startBackgroundAudio()
    await startBackgroundAudio()
    expect(captureMessage).toHaveBeenCalledTimes(1)
    expect(captureMessage.mock.calls[0][1]).toMatchObject({
      level: 'warning',
      fingerprint: ['android-foreground-refused'],
    })
  })

  it('reports nothing when playback is in the foreground', async () => {
    const { startBackgroundAudio } = await import('./native')
    start.mockResolvedValue({ foreground: true })
    await startBackgroundAudio()
    expect(captureMessage).not.toHaveBeenCalled()
  })

  it('reports nothing, and does not throw, when the plugin rejects or answers an older shape', async () => {
    const { startBackgroundAudio } = await import('./native')
    start.mockRejectedValueOnce(new Error('not implemented'))
    await startBackgroundAudio()
    start.mockResolvedValueOnce(undefined) // a shell built before the plugin returned anything
    await startBackgroundAudio()
    expect(captureMessage).not.toHaveBeenCalled()
  })
})
