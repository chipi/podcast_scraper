/**
 * The player store, mirrored into the ANDROID media session (operator 2026-10-05: no lock-screen
 * controls and no output button on Android — the WebView never turned `navigator.mediaSession` into
 * them). The native side is mocked; what is pinned here is the store's half of the contract.
 */
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const native = vi.hoisted(() => ({
  startBackgroundAudio: vi.fn(async () => {}),
  pauseBackgroundAudio: vi.fn(async () => {}),
  stopBackgroundAudio: vi.fn(async () => {}),
  updateNativeMedia: vi.fn(async () => {}),
  onNativeMediaAction: vi.fn(),
  canShowOutputSwitcher: vi.fn(async () => false),
  showOutputSwitcher: vi.fn(async () => true),
}))
vi.mock('../services/native', () => native)

import { usePlayerStore } from './player'

function stubAudio() {
  const listeners: Record<string, (() => void)[]> = {}
  const el = {
    paused: true,
    currentTime: 0,
    duration: 600,
    playbackRate: 1,
    volume: 1,
    src: '',
    preload: '',
    readyState: 4,
    style: {},
    setAttribute: vi.fn(),
    removeAttribute: vi.fn(),
    play: vi.fn(function (this: { paused: boolean }) {
      this.paused = false
      return Promise.resolve()
    }),
    pause: vi.fn(function (this: { paused: boolean }) {
      this.paused = true
    }),
    addEventListener: vi.fn((k: string, h: () => void) => {
      ;(listeners[k] ??= []).push(h)
    }),
    emit: (k: string) => listeners[k]?.forEach((h) => h()),
  }
  vi.stubGlobal('Audio', vi.fn(function () { return el }))
  vi.spyOn(document.body, 'appendChild').mockImplementation((n) => n)
  return el
}

describe('player store → Android media session', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    for (const f of Object.values(native)) f.mockClear()
    native.canShowOutputSwitcher.mockResolvedValue(false)
  })

  it('PAUSE keeps the lock-screen controls (pause, not stop)', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.load({ slug: 'ep-1', url: 'https://x/a.mp3', title: 'An Episode' })
    el.emit('play')
    native.stopBackgroundAudio.mockClear()
    el.emit('pause')
    expect(native.pauseBackgroundAudio).toHaveBeenCalled()
    expect(native.stopBackgroundAudio).not.toHaveBeenCalled()
  })

  it('the episode reaches the native session: title, show, artwork', () => {
    stubAudio()
    const p = usePlayerStore()
    p.setMetadata({ title: 'Ep', artist: 'Show', artworkUrl: 'https://x/a.png' })
    expect(native.updateNativeMedia).toHaveBeenCalledWith(
      expect.objectContaining({ title: 'Ep', artist: 'Show', artworkUrl: 'https://x/a.png' }),
    )
  })

  it('a lock-screen Pause and a lock-screen seek drive the player', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.load({ slug: 'ep-1', url: 'https://x/a.mp3', title: 'An Episode' })
    p.setMetadata({ title: 'Ep' })
    const relay = native.onNativeMediaAction.mock.calls[0]?.[0] as (a: object) => void
    expect(relay, 'the store never subscribed to native actions').toBeTypeOf('function')
    el.paused = false
    relay({ action: 'pause' })
    expect(el.pause).toHaveBeenCalled()
    relay({ action: 'seekto', seekTime: 120 })
    expect(el.currentTime).toBe(120)
  })

  it('on Android 14+ the route button exists and opens the system output switcher', async () => {
    native.canShowOutputSwitcher.mockResolvedValue(true)
    stubAudio()
    const p = usePlayerStore()
    p.load({ slug: 'ep-1', url: 'https://x/a.mp3', title: 'An Episode' })
    await vi.waitFor(() => expect(p.routeAvailable).toBe(true))
    p.showRoutePicker()
    expect(native.showOutputSwitcher).toHaveBeenCalled()
  })
})
