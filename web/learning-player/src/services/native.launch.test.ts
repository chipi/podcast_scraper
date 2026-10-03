import { beforeEach, describe, expect, it, vi } from 'vitest'

// A sign-in link can reach the app two ways: while it is running (`appUrlOpen`), or by LAUNCHING
// it (`getLaunchUrl`, for which `appUrlOpen` does not fire). The second is the ordinary magic-link
// case — the person taps "Sign in" in Mail with the app closed — and was silently dropped.

let launchUrl: string | null = null
let urlOpen: ((e: { url: string }) => void) | null = null

vi.mock('@capacitor/core', () => ({
  Capacitor: { isNativePlatform: () => true, getPlatform: () => 'android' },
  CapacitorCookies: { setCookie: vi.fn(async () => {}) },
  registerPlugin: () => ({}),
}))
vi.mock('@capacitor/app', () => ({
  App: {
    addListener: vi.fn(async (_name: string, cb: (e: { url: string }) => void) => {
      urlOpen = cb
      return { remove: async () => {} }
    }),
    getLaunchUrl: vi.fn(async () => (launchUrl ? { url: launchUrl } : undefined)),
  },
}))
vi.mock('@capacitor/browser', () => ({ Browser: { close: vi.fn(async () => {}) } }))
vi.mock('@capacitor/preferences', () => ({
  Preferences: {
    get: vi.fn(async () => ({ value: null })),
    set: vi.fn(async () => {}),
    remove: vi.fn(async () => {}),
  },
}))
vi.mock('./api', () => ({ setAuthToken: vi.fn() }))

const { initNativeAuth } = await import('./native')
const api = await import('./api')

beforeEach(() => {
  launchUrl = null
  urlOpen = null
  vi.mocked(api.setAuthToken).mockClear()
})

describe('initNativeAuth', () => {
  it('signs in from a link that LAUNCHED the app', async () => {
    launchUrl = 'closelistening://auth#token=cold.tok&new=1'
    const onAuthed = vi.fn()
    await initNativeAuth(onAuthed)
    expect(api.setAuthToken).toHaveBeenCalledWith('cold.tok')
    expect(onAuthed).toHaveBeenCalledWith({ isNew: true })
  })

  it('signs in from a link that arrives while running', async () => {
    const onAuthed = vi.fn()
    await initNativeAuth(onAuthed)
    expect(onAuthed).not.toHaveBeenCalled()
    urlOpen!({ url: 'closelistening://auth#token=warm.tok' })
    expect(api.setAuthToken).toHaveBeenCalledWith('warm.tok')
    expect(onAuthed).toHaveBeenCalledWith({ isNew: false })
  })

  it('handles a URL delivered through BOTH paths once', async () => {
    launchUrl = 'closelistening://auth#token=same.tok&new=1'
    const onAuthed = vi.fn()
    await initNativeAuth(onAuthed)
    urlOpen!({ url: launchUrl })
    expect(onAuthed).toHaveBeenCalledTimes(1)
  })

  it('ignores a launch URL that is not a sign-in callback', async () => {
    launchUrl = 'closelistening://episode/some-slug'
    const onAuthed = vi.fn()
    await initNativeAuth(onAuthed)
    expect(onAuthed).not.toHaveBeenCalled()
    expect(api.setAuthToken).not.toHaveBeenCalled()
  })
})
