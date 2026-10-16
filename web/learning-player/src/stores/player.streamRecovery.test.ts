import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const originOfStreamUrl = vi.fn()
const refreshStreamUrl = vi.fn()
vi.mock('../services/api', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../services/api')>()),
  originOfStreamUrl: (u: string) => originOfStreamUrl(u),
  refreshStreamUrl: (u: string) => refreshStreamUrl(u),
}))

import { usePlayerStore } from './player'

/**
 * RFC-129: playback plays a host's final redirected URL, which is signed and expires (75 min seen).
 * After a long pause the next request fails; the player resolves the original again, once, and goes
 * on from where it was.
 */

function stubAudio() {
  const listeners: Record<string, (() => void)[]> = {}
  const el = {
    paused: true,
    currentTime: 0,
    duration: 1800,
    readyState: 4,
    playbackRate: 1,
    volume: 1,
    src: '',
    preload: '',
    error: { code: 2 },
    style: {} as CSSStyleDeclaration,
    setAttribute: vi.fn(),
    play: vi.fn(function (this: { paused: boolean }) {
      this.paused = false
      return Promise.resolve()
    }),
    pause: vi.fn(function (this: { paused: boolean }) {
      this.paused = true
    }),
    removeAttribute: vi.fn(),
    addEventListener: vi.fn((k: string, h: () => void) => {
      ;(listeners[k] ??= []).push(h)
    }),
    removeEventListener: vi.fn(),
    load: vi.fn(),
  }
  vi.stubGlobal(
    'Audio',
    vi.fn(function (this: unknown) {
      return el
    }),
  )
  vi.spyOn(document.body, 'appendChild').mockImplementation((n) => n)
  return el
}

async function drain(): Promise<void> {
  for (let i = 0; i < 6; i++) await Promise.resolve()
}

describe('an expired stream URL (#2359)', () => {
  let el: ReturnType<typeof stubAudio>
  let p: ReturnType<typeof usePlayerStore>
  beforeEach(() => {
    originOfStreamUrl.mockReset()
    refreshStreamUrl.mockReset()
    setActivePinia(createPinia())
    p = usePlayerStore()
    el = stubAudio()
    p.load({ slug: 'ep-1', url: 'https://cdn/final-1.mp3', title: 'An Episode' })
    el.currentTime = 1500
    p.onTimeUpdate()
    p.playing = true
  })
  afterEach(() => vi.unstubAllGlobals())

  it('resolves the original again and carries on from the same place', async () => {
    originOfStreamUrl.mockReturnValue('https://feed/ep.mp3')
    refreshStreamUrl.mockResolvedValue('https://cdn/final-2.mp3')
    p.onError()
    await drain()
    expect(refreshStreamUrl).toHaveBeenCalledWith('https://feed/ep.mp3')
    expect(el.src).toBe('https://cdn/final-2.mp3')
    expect(p.streamUrl).toBe('https://cdn/final-2.mp3')
    expect(el.currentTime).toBe(1500)
    expect(el.play).toHaveBeenCalled()
    expect(p.audioError).toBe(false)
  })

  it('tries once per URL: a second failure on the fresh one shows the error', async () => {
    originOfStreamUrl.mockReturnValue('https://feed/ep.mp3')
    refreshStreamUrl.mockResolvedValue('https://cdn/final-2.mp3')
    p.onError()
    await drain()
    originOfStreamUrl.mockReturnValue('https://feed/ep.mp3')
    refreshStreamUrl.mockResolvedValue('https://cdn/final-2.mp3')
    p.onError()
    await drain()
    expect(p.audioError).toBe(true)
  })

  it('shows the error at once for a URL it did not resolve', async () => {
    originOfStreamUrl.mockReturnValue(null)
    p.onError()
    await drain()
    expect(refreshStreamUrl).not.toHaveBeenCalled()
    expect(p.audioError).toBe(true)
  })
})
