import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PLAYBACK_RATES } from '../player/transcriptSync'
import { usePlayerStore } from './player'

/** Minimal fake <audio> — the store only touches these props/methods. */
function fakeAudio(over: Partial<HTMLAudioElement> = {}) {
  return {
    paused: true,
    currentTime: 0,
    duration: 0,
    playbackRate: 1,
    play: vi.fn(function (this: { paused: boolean }) {
      this.paused = false
      return Promise.resolve()
    }),
    pause: vi.fn(function (this: { paused: boolean }) {
      this.paused = true
    }),
    ...over,
  } as unknown as HTMLAudioElement
}

describe('player store', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('bind() applies the current rate to the element', () => {
    const p = usePlayerStore()
    p.setRate(1.5)
    const el = fakeAudio()
    p.bind(el)
    expect(el.playbackRate).toBe(1.5)
  })

  it('play/pause + timeupdate + durationchange sinks reflect element state', () => {
    const p = usePlayerStore()
    const el = fakeAudio({ currentTime: 42, duration: 100 })
    p.bind(el)
    p.onPlay()
    expect(p.playing).toBe(true)
    p.onPause()
    expect(p.playing).toBe(false)
    p.onTimeUpdate()
    expect(p.currentTime).toBe(42)
    p.onDurationChange()
    expect(p.duration).toBe(100)
  })

  it('toggle() plays when paused and pauses when playing', () => {
    const p = usePlayerStore()
    const el = fakeAudio({ paused: true })
    p.bind(el)
    p.toggle()
    expect(el.play).toHaveBeenCalledOnce()
    expect(el.paused).toBe(false)
    p.toggle()
    expect(el.pause).toHaveBeenCalledOnce()
    expect(el.paused).toBe(true)
  })

  it('seek() clamps to [0, duration]; skip() is relative', () => {
    const p = usePlayerStore()
    const el = fakeAudio({ currentTime: 50, duration: 100 })
    p.bind(el)
    p.onDurationChange()
    p.seek(-5)
    expect(el.currentTime).toBe(0)
    p.seek(999)
    expect(el.currentTime).toBe(100)
    el.currentTime = 50
    p.onTimeUpdate() // store currentTime = 50 (skip derives from store state, like the browser)
    p.skip(15)
    expect(el.currentTime).toBe(65)
    p.onTimeUpdate() // browser fires timeupdate after the seek → store catches up
    p.skip(-30)
    expect(el.currentTime).toBe(35)
  })

  it('cycleRate() advances through PLAYBACK_RATES and applies to the element', () => {
    const p = usePlayerStore()
    const el = fakeAudio()
    p.bind(el)
    p.setRate(PLAYBACK_RATES[0])
    p.cycleRate()
    expect(p.rate).toBe(PLAYBACK_RATES[1])
    expect(el.playbackRate).toBe(PLAYBACK_RATES[1])
    // wraps around from the last rate back to the first
    p.setRate(PLAYBACK_RATES[PLAYBACK_RATES.length - 1])
    p.cycleRate()
    expect(p.rate).toBe(PLAYBACK_RATES[0])
  })

  it('onError sets audioError; resetForLoad clears transient state', () => {
    const p = usePlayerStore()
    const el = fakeAudio({ currentTime: 30, duration: 60 })
    p.bind(el)
    p.onPlay()
    p.onTimeUpdate()
    p.onDurationChange()
    p.onError()
    expect(p.audioError).toBe(true)
    p.resetForLoad()
    expect(p.playing).toBe(false)
    expect(p.currentTime).toBe(0)
    expect(p.duration).toBe(0)
    expect(p.audioError).toBe(false)
  })
})

describe('player store — MediaSession (#1308)', () => {
  let ms: {
    metadata: unknown
    playbackState: string
    setActionHandler: ReturnType<typeof vi.fn>
    setPositionState: ReturnType<typeof vi.fn>
  }
  beforeEach(() => {
    setActivePinia(createPinia())
    ms = { metadata: null, playbackState: 'none', setActionHandler: vi.fn(), setPositionState: vi.fn() }
    ;(navigator as unknown as { mediaSession: unknown }).mediaSession = ms
    ;(globalThis as unknown as { MediaMetadata: unknown }).MediaMetadata = class {
      constructor(public init: Record<string, unknown>) {}
    }
  })
  afterEach(() => {
    delete (navigator as unknown as { mediaSession?: unknown }).mediaSession
    delete (globalThis as unknown as { MediaMetadata?: unknown }).MediaMetadata
  })

  it('setMetadata sets lock-screen metadata + wires the action handlers', () => {
    const p = usePlayerStore()
    p.setMetadata({ title: 'Ep 1', artist: 'The Show', artworkUrl: 'a.png' })
    expect((ms.metadata as { init: { title: string } }).init.title).toBe('Ep 1')
    const actions = ms.setActionHandler.mock.calls.map((c) => c[0])
    expect(actions).toEqual(
      expect.arrayContaining(['play', 'pause', 'seekbackward', 'seekforward', 'seekto', 'previoustrack', 'nexttrack']),
    )
  })

  it('reflects playbackState + positionState from element events', () => {
    const p = usePlayerStore()
    p.bind(fakeAudio({ currentTime: 10, duration: 120 }))
    p.onPlay()
    expect(ms.playbackState).toBe('playing')
    p.onPause()
    expect(ms.playbackState).toBe('paused')
    p.onDurationChange()
    p.onTimeUpdate()
    const last = ms.setPositionState.mock.calls.at(-1)![0]
    expect(last.duration).toBe(120)
    expect(last.position).toBe(10)
  })

  it('nexttrack / previoustrack invoke the registered skip handlers', () => {
    const p = usePlayerStore()
    const next = vi.fn()
    const prev = vi.fn()
    p.setSkipHandlers({ next, prev })
    const byAction: Record<string, (d?: unknown) => void> = {}
    for (const [a, h] of ms.setActionHandler.mock.calls) byAction[a as string] = h as (d?: unknown) => void
    byAction['nexttrack']()
    byAction['previoustrack']()
    expect(next).toHaveBeenCalledOnce()
    expect(prev).toHaveBeenCalledOnce()
  })
})
