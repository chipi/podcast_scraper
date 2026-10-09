import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PLAYBACK_RATES } from '../player/transcriptSync'
import { usePlayerStore } from './player'

/**
 * Minimal fake <audio>. The store CONSTRUCTS its own element now (#1587), so the constructor is
 * stubbed rather than an element being handed in — that inversion is the whole point of the change:
 * the element's lifetime is the store's, not a view's.
 */
function fakeAudio(over: Partial<HTMLAudioElement> = {}) {
  const listeners: Record<string, ((e?: unknown) => void)[]> = {}
  return {
    paused: true,
    currentTime: 0,
    duration: 0,
    playbackRate: 1,
    volume: 1,
    src: '',
    preload: '',
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
    /** Test helper: fire a listener the store attached. */
    __emit: (k: string) => listeners[k]?.forEach((h) => h()),
    ...over,
  } as unknown as HTMLAudioElement & { __emit: (k: string) => void }
}

/** Install the fake as the global Audio constructor and return the instance the store will build. */
function stubAudio(over: Partial<HTMLAudioElement> = {}) {
  const el = fakeAudio(over)
  // Must be constructible — the store does `new Audio()`. An arrow function is not.
  vi.stubGlobal(
    'Audio',
    vi.fn(function (this: unknown) {
      return el
    }),
  )
  // The store appends the element to <body> so it stays inspectable (full-listen.spec and the
  // tier-3 walk both use document.querySelector('audio')). This fake is not a real Node, so the
  // append is neutralised here rather than weakened in the store.
  vi.spyOn(document.body, 'appendChild').mockImplementation((n) => n)
  return el
}

/** Load an episode so the store constructs + wires its element. */
function loaded(p: ReturnType<typeof usePlayerStore>, el: HTMLAudioElement) {
  p.load({ slug: 'ep-1', url: 'https://x/a.mp3', title: 'An Episode' })
  return el
}

/** onEnded / playNext are async (they await the resolver) — drain their microtasks. */
async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 4; i++) await Promise.resolve()
}

describe('player store', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('setVolumeLevel applies the factor to the element and persists it', () => {
    localStorage.removeItem('lp.volume')
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.setVolumeLevel('low')
    expect(el.volume).toBe(0.4)
    expect(p.volumeLevel).toBe('low')
    expect(localStorage.getItem('lp.volume')).toBe('low')
    p.setVolumeLevel('high')
    expect(el.volume).toBe(1)
  })

  it('restores the persisted volume on boot and applies it to the freshly built element', () => {
    // A returning listener keeps their chosen level across app restarts: the store reads lp.volume at
    // init and ensureElement() must apply it, or the first-built element would play at full volume.
    localStorage.setItem('lp.volume', 'medium')
    const el = stubAudio()
    const p = usePlayerStore()
    expect(p.volumeLevel).toBe('medium')
    loaded(p, el)
    expect(el.volume).toBe(0.7)
  })

  it('load() builds an element, applies the rate, and records the episode', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.setRate(1.5)
    loaded(p, el)
    expect(el.playbackRate).toBe(1.5)
    expect(el.src).toBe('https://x/a.mp3')
    expect(p.currentSlug).toBe('ep-1')
    expect(p.currentTitle).toBe('An Episode')
  })

  it('re-loading the SAME episode does not restart it (#1587)', () => {
    // Returning to the player page mid-listen must not reset playback — that would replace one
    // annoyance with a worse one.
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.onDurationChange()
    ;(el as unknown as { currentTime: number }).currentTime = 42
    p.onTimeUpdate()
    expect(p.currentTime).toBe(42)

    p.load({ slug: 'ep-1', url: 'https://x/a.mp3' })
    expect(p.currentTime).toBe(42)
  })

  it('the element outlives any view — one element across episodes', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    const first = p.el
    p.load({ slug: 'ep-2', url: 'https://x/b.mp3' })
    expect(p.el).toBe(first)
    expect(globalThis.Audio).toHaveBeenCalledTimes(1)
  })

  it('clear() stops and forgets the episode', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.clear()
    expect(el.pause).toHaveBeenCalled()
    expect(p.currentSlug).toBeNull()
    expect(p.playing).toBe(false)
  })

  it('play/pause + timeupdate + durationchange sinks reflect element state', () => {
    const p = usePlayerStore()
    const el = stubAudio({ currentTime: 42, duration: 100 })
    loaded(p, el)
    p.onPlay()
    expect(p.playing).toBe(true)
    p.onPause()
    expect(p.playing).toBe(false)
    p.onTimeUpdate()
    expect(p.currentTime).toBe(42)
    p.onDurationChange()
    expect(p.duration).toBe(100)
  })

  /**
   * Offline, a downloaded file often makes the element report NO duration at all — so the transport
   * showed `0:00` as the total and the scrub bar sat at zero for a whole episode while the elapsed
   * time climbed (operator 2026-09-19, listening offline at 1:02:57 of an episode reading 0:00).
   */
  describe('duration falls back to the episode metadata', () => {
    it('uses the hint while the element reports nothing', () => {
      const p = usePlayerStore()
      const el = stubAudio({ duration: 0 })
      p.load({ slug: 'ep-1', url: 'file:///a.mp3', title: 'An Episode', durationSeconds: 3600 })
      p.onDurationChange()
      expect(el.duration).toBe(0)
      expect(p.duration).toBe(3600)
    })

    it('the element WINS once it knows — the file is the truth about the file', () => {
      const p = usePlayerStore()
      const el = stubAudio({ duration: 3550 })
      p.load({ slug: 'ep-1', url: 'file:///a.mp3', title: 'An Episode', durationSeconds: 3600 })
      p.onDurationChange()
      expect(el.duration).toBe(3550)
      expect(p.duration).toBe(3550)
    })

    it('a later load() without a hint does not keep the previous episode length', () => {
      const p = usePlayerStore()
      stubAudio({ duration: 0 })
      p.load({ slug: 'ep-1', url: 'file:///a.mp3', title: 'One', durationSeconds: 3600 })
      expect(p.duration).toBe(3600)
      p.load({ slug: 'ep-2', url: 'file:///b.mp3', title: 'Two' })
      expect(p.duration).toBe(0)
    })
  })

  it('toggle() plays when paused and pauses when playing', () => {
    const p = usePlayerStore()
    const el = stubAudio({ paused: true })
    loaded(p, el)
    p.toggle()
    expect(el.play).toHaveBeenCalledOnce()
    expect(el.paused).toBe(false)
    p.toggle()
    expect(el.pause).toHaveBeenCalledOnce()
    expect(el.paused).toBe(true)
  })

  it('resumeAfterInterruption() plays again without counting a new listen', async () => {
    // The mic pausing an episode and dictation handing it back is not the reader pressing Play.
    const analytics = await import('../services/analytics')
    const track = vi.spyOn(analytics, 'track')
    const p = usePlayerStore()
    const el = stubAudio({ paused: true })
    loaded(p, el)
    p.resumeAfterInterruption()
    expect(el.play).toHaveBeenCalledOnce()
    expect(track).not.toHaveBeenCalledWith('play_start', expect.anything())
    p.play()
    expect(track).toHaveBeenCalledWith('play_start', expect.anything())
  })

  it('seek() clamps to [0, duration]; skip() is relative', () => {
    const p = usePlayerStore()
    const el = stubAudio({ currentTime: 50, duration: 100 })
    loaded(p, el)
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
    const el = stubAudio()
    loaded(p, el)
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
    const el = stubAudio({ currentTime: 30, duration: 60 })
    loaded(p, el)
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
      init: Record<string, unknown>
      constructor(init: Record<string, unknown>) { this.init = init }
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
    loaded(p, stubAudio({ currentTime: 10, duration: 120 }))
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

  // --- auto-advance (#1587), rebuilt after the fable-5 review ---

  /** onEnded is async now (it awaits the resolver), so let its microtasks drain. */
  const settle = async () => {
    for (let i = 0; i < 4; i++) await Promise.resolve()
  }

  it('resolves what plays next AT the end, not when the episode started', async () => {
    // The first version cached the answer at load. That ignored every input made while listening —
    // "Play next", a reorder, a first queue item — and mid-listen is when all of them happen.
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)

    let answer: { slug: string; url: string; title: string } | null = null
    p.setAdvanceResolver(async () => answer)

    // Queued only AFTER playback started — the case that used to end in silence.
    answer = { slug: 'b', url: 'https://x/b.mp3', title: 'B' }
    el.__emit('ended')
    await settle()

    expect(p.currentSlug).toBe('b')
    expect(el.src).toBe('https://x/b.mp3')
  })

  it('carries the next title and artwork, so the mini-player is not stuck on "Loading…"', async () => {
    // Auto-advance happens with NO view mounted, so nothing else can supply these. Without them the
    // mini-player renders its loading fallback for the whole episode and the lock screen keeps
    // showing the previous one.
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.setAdvanceResolver(async () => ({
      slug: 'b',
      url: 'https://x/b.mp3',
      title: 'Episode B',
      artwork: 'https://x/b.png',
    }))

    el.__emit('ended')
    await settle()

    expect(p.currentTitle).toBe('Episode B')
    expect(p.currentArtwork).toBe('https://x/b.png')
  })

  it('stops cleanly when nothing is queued', async () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.setAdvanceResolver(async () => null)

    el.__emit('ended')
    await settle()

    expect(p.currentSlug).toBe('ep-1')
    expect(p.playing).toBe(false)
  })
})

describe('position persistence', () => {
  beforeEach(() => setActivePinia(createPinia()))

  /** Drive `timeupdate` at a given element time, past the throttle window. */
  function tick(p: ReturnType<typeof usePlayerStore>, el: HTMLAudioElement, at: number) {
    ;(el as unknown as { currentTime: number }).currentTime = at
    vi.advanceTimersByTime(11_000)
    ;(el as unknown as { __emit: (k: string) => void }).__emit('timeupdate')
  }

  it('writes the slug and the time that belong to the SAME episode', () => {
    // The bug this replaces: PlayerView paired ITS route's slug with the STORE's currentTime. When
    // an episode auto-advanced while the user sat on the player page, every save for the new
    // episode landed on the old episode's record — every 10s, for the whole episode.
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    const saves: [string, number][] = []
    p.setPositionPersister((slug, seconds) => saves.push([slug, seconds]))

    loaded(p, el)
    tick(p, el, 30)
    // Auto-advance: the element is now playing a DIFFERENT episode, with no navigation.
    p.load({ slug: 'ep-2', url: 'https://x/b.mp3', title: 'B' })
    tick(p, el, 7)

    expect(saves).toContainEqual(['ep-1', 30])
    expect(saves).toContainEqual(['ep-2', 7])
    // The pair that used to be written: ep-1's record carrying ep-2's timeline.
    expect(saves.filter(([slug, s]) => slug === 'ep-1' && s === 7)).toEqual([])
    vi.useRealTimers()
  })

  it('flushes the outgoing episode before its identity is overwritten', () => {
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    const saves: [string, number][] = []
    p.setPositionPersister((slug, seconds) => saves.push([slug, seconds]))

    loaded(p, el)
    tick(p, el, 30)
    ;(el as unknown as { currentTime: number }).currentTime = 44 // 14s further, inside the window
    p.load({ slug: 'ep-2', url: 'https://x/b.mp3' })

    expect(saves).toContainEqual(['ep-1', 44]) // not lost to the throttle
    vi.useRealTimers()
  })

  it('saves with no view mounted — the store is the only thing that has to be alive', () => {
    // Persistence used to live in PlayerView, so listening via the mini-player (the entire point of
    // #1587) recorded nothing at all: an hour from Home left the resume point untouched.
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    const saves: [string, number][] = []
    p.setPositionPersister((slug, seconds) => saves.push([slug, seconds]))

    loaded(p, el)
    tick(p, el, 120)

    expect(saves).toEqual([['ep-1', 120]])
    vi.useRealTimers()
  })

  it('flushes on pause, without waiting for the throttle window', () => {
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    const saves: [string, number][] = []
    p.setPositionPersister((slug, seconds) => saves.push([slug, seconds]))

    loaded(p, el)
    tick(p, el, 30)
    ;(el as unknown as { currentTime: number }).currentTime = 33
    el.__emit('pause')

    expect(saves).toContainEqual(['ep-1', 33])
    vi.useRealTimers()
  })

  it('throttles the timeupdate firehose', () => {
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    const saves: [string, number][] = []
    p.setPositionPersister((slug, seconds) => saves.push([slug, seconds]))

    loaded(p, el)
    for (let i = 1; i <= 20; i += 1) {
      ;(el as unknown as { currentTime: number }).currentTime = i
      vi.advanceTimersByTime(250) // ~4 timeupdates a second, as a real element fires
      el.__emit('timeupdate')
    }
    expect(saves.length).toBeLessThanOrEqual(1)
    vi.useRealTimers()
  })

  it('is inert until the shell supplies a writer', () => {
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    expect(() => tick(p, el, 30)).not.toThrow()
    vi.useRealTimers()
  })
})

describe('a refused play() is recorded, not dropped', () => {
  beforeEach(() => setActivePinia(createPinia()))

  /** onEnded and the play() catch are both async — let their microtasks drain. */
  const settle = async () => {
    for (let i = 0; i < 4; i++) await Promise.resolve()
  }

  it('flags a dead source when auto-advance cannot play it', async () => {
    // load() clears audioError via resetForLoad, so advancing INTO a broken episode wiped the only
    // flag and then swallowed the play() rejection. Result: silence, with the mini-player showing
    // the new title, paused, and nothing anywhere saying why.
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    ;(el as unknown as { play: () => Promise<void> }).play = () =>
      Promise.reject(new DOMException('no', 'NotSupportedError'))
    p.setAdvanceResolver(async () => ({ slug: 'ep-2', url: 'https://x/b.mp3', title: 'B' }))

    el.__emit('ended')
    await settle()

    expect(p.currentSlug).toBe('ep-2')
    expect(p.audioError).toBe(true)
  })

  it('flags a dead source when the user presses play', async () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    ;(el as unknown as { play: () => Promise<void> }).play = () =>
      Promise.reject(new DOMException('no', 'NotSupportedError'))

    p.toggle()
    await settle()

    expect(p.audioError).toBe(true)
  })

  it('does NOT flag the autoplay policy — that is a gesture request, not a broken episode', async () => {
    // Telling someone their audio is unavailable when the browser is merely asking for a tap would
    // be a lie, and one that survives until the next load().
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    ;(el as unknown as { play: () => Promise<void> }).play = () =>
      Promise.reject(new DOMException('gesture required', 'NotAllowedError'))

    p.toggle()
    await settle()

    expect(p.audioError).toBe(false)
  })
})

describe('finishing an episode', () => {
  beforeEach(() => setActivePinia(createPinia()))

  const settle = async () => {
    for (let i = 0; i < 4; i++) await Promise.resolve()
  }

  function persisted(p: ReturnType<typeof usePlayerStore>) {
    const saves: [string, number, boolean][] = []
    p.setPositionPersister((slug, seconds, finished) => saves.push([slug, seconds, finished]))
    return saves
  }

  it('records the finish BEFORE auto-advance overwrites the episode identity', async () => {
    // Nothing marked an episode finished, so the last cadence save left it parked seconds from the
    // end: it sat in "Continue listening" forever, and re-opening it resumed at end-epsilon and
    // instantly auto-advanced away again.
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    const saves = persisted(p)
    ;(el as unknown as { currentTime: number }).currentTime = 1800
    p.setAdvanceResolver(async () => ({ slug: 'ep-2', url: 'https://x/b.mp3' }))

    el.__emit('ended')
    await settle()

    expect(saves.some(([slug, , finished]) => slug === 'ep-1' && finished)).toBe(true)
  })

  it('counts a skipped outro as finished — `ended` never fires for it', async () => {
    // The reason the threshold exists alongside the flag: skipping the last minute is a normal way
    // to finish an episode, and those would otherwise live in Continue forever.
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    const saves = persisted(p)
    el.__emit('durationchange')
    ;(el as unknown as { duration: number }).duration = 1000
    el.__emit('durationchange')
    ;(el as unknown as { currentTime: number }).currentTime = 960 // 96%
    vi.advanceTimersByTime(11_000)
    el.__emit('timeupdate')

    expect(saves.at(-1)).toEqual(['ep-1', 960, true])
    vi.useRealTimers()
  })

  it('does not call an episode finished part-way through', async () => {
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    const saves = persisted(p)
    ;(el as unknown as { duration: number }).duration = 1000
    el.__emit('durationchange')
    ;(el as unknown as { currentTime: number }).currentTime = 700 // 70%
    vi.advanceTimersByTime(11_000)
    el.__emit('timeupdate')

    expect(saves.at(-1)).toEqual(['ep-1', 700, false])
    vi.useRealTimers()
  })

  it('an unknown duration never reads as finished', async () => {
    // duration 0 would make at/d NaN or Infinity; neither may be treated as "reached the end".
    vi.useFakeTimers()
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    const saves = persisted(p)
    ;(el as unknown as { currentTime: number }).currentTime = 42
    vi.advanceTimersByTime(11_000)
    el.__emit('timeupdate')

    expect(saves.at(-1)).toEqual(['ep-1', 42, false])
    vi.useRealTimers()
  })

  // #1905 — a downloaded episode plays from disk. The resolver is injected by the shell so this
  // store keeps knowing nothing about downloads or the API.
  it('load() prefers the injected local source over the origin URL', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.setSourceResolver((slug) => (slug === 'a' ? 'capacitor-file:///local/a.mp3' : null))
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('capacitor-file:///local/a.mp3')
  })

  it('load() falls back to the origin URL when nothing is downloaded', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.setSourceResolver(() => null)
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('https://x/a.mp3')
  })

  it('load() streams normally when no resolver was ever injected', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('https://x/a.mp3')
  })

  // Repro (test-ios phase 2, 2026-10-04): the cold-launch restore loads the last episode BEFORE the
  // shell injects the local resolver (and before the downloads registry is read), so it is pinned
  // to the origin URL. Opening the downloaded copy then re-loads the SAME slug, which used to be a
  // no-op — offline that left "Couldn't load the audio from the source" on a downloaded episode.
  it('re-loading the same episode switches to a local copy that became known since', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('https://x/a.mp3')
    el.currentTime = 37
    p.setSourceResolver((slug) => (slug === 'a' ? 'capacitor-file:///local/a.mp3' : null))
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('capacitor-file:///local/a.mp3')
    // The position survives the source switch: applied once the new source has metadata.
    ;(el as unknown as { __emit: (k: string) => void }).__emit('loadedmetadata')
    expect(el.currentTime).toBe(37)
  })

  // The device half of the same repro: offline, the pinned origin URL ERRORS before the user opens
  // the downloaded copy, setting audioError — which only resetForLoad() cleared. Switching the source
  // without clearing it left the view on "Couldn't load the audio from the source" with no Play.
  it('switching to the local copy clears the error the dead origin URL raised', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    p.onError()
    expect(p.audioError).toBe(true)
    p.setSourceResolver((slug) => (slug === 'a' ? 'capacitor-file:///local/a.mp3' : null))
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('capacitor-file:///local/a.mp3')
    expect(p.audioError).toBe(false)
  })

  it('re-loading the same episode does NOT touch a source that is already the local copy', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    p.setSourceResolver(() => 'capacitor-file:///local/a.mp3')
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    el.currentTime = 12
    p.load({ slug: 'a', url: 'https://x/a.mp3', title: 'A', artwork: null })
    expect(el.src).toBe('capacitor-file:///local/a.mp3')
    expect(el.currentTime).toBe(12)
  })
})

/**
 * A listen is PLAYBACK, not a page view (#1925 review C6). PlayerView calls `load()` on mount, so
 * arming-on-load meant browsing episodes without pressing play logged a listen for each one.
 */
describe('listen logging', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('does not log on load alone — opening a page is not listening', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    const logged: string[] = []
    p.setListenLogger((slug) => logged.push(slug))
    loaded(p, el)
    expect(logged).toEqual([])
  })

  it('logs once on the first play, and not again on resume', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    const logged: string[] = []
    p.setListenLogger((slug) => logged.push(slug))
    loaded(p, el)
    el.__emit('play')
    el.__emit('pause')
    el.__emit('play')
    expect(logged).toEqual(['ep-1'])
  })

  it('logs the next episode too — auto-advance loads and plays with no view mounted', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    const logged: string[] = []
    p.setListenLogger((slug) => logged.push(slug))
    loaded(p, el)
    el.__emit('play')
    p.load({ slug: 'ep-2', url: 'https://x/b.mp3', title: 'Next' })
    el.__emit('play')
    expect(logged).toEqual(['ep-1', 'ep-2'])
  })

  it("close() keeps the listener's place, then forgets the episode (mini-player ✕, operator 2026-10-07)", () => {
    const el = stubAudio()
    const p = usePlayerStore()
    const saves: [string, number][] = []
    p.setPositionPersister((slug, seconds) => saves.push([slug, seconds]))
    loaded(p, el)
    ;(el as unknown as { currentTime: number }).currentTime = 42
    el.__emit('timeupdate')
    saves.length = 0
    p.close()
    // Saved BEFORE the slug is gone: the pause event clear() causes would arrive with no episode.
    expect(saves).toContainEqual(['ep-1', 42])
    expect(p.currentSlug).toBeNull()
    expect(p.nowPlaying()).toBeNull()
  })

  it('forgets an armed-but-unplayed episode on clear — sign-out is not a listen', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    const logged: string[] = []
    p.setListenLogger((slug) => logged.push(slug))
    loaded(p, el)
    p.clear()
    el.__emit('play')
    expect(logged).toEqual([])
  })

  // --- justFinished: the finish signal the recap panel watches (RFC-122 #2038) ---

  it('justFinished names the episode on `ended`', () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    expect(p.justFinished).toBeNull()
    el.__emit('ended') // onEnded → savePosition(true)
    expect(p.justFinished).toBe('ep-1')
  })

  it('justFinished fires past the 95% threshold even without an `ended` event', () => {
    // Skipping the outro is a normal way to finish; the fraction catches it. And it is set even with
    // no persister (signed-out / tests) — finishing is a playback fact, not a persistence one.
    const el = stubAudio({ currentTime: 96, duration: 100 })
    const p = usePlayerStore()
    loaded(p, el)
    p.onDurationChange()
    p.savePosition()
    expect(p.justFinished).toBe('ep-1')
  })

  it('justFinished does NOT fire past 95% while still PLAYING, then does on pause (B3)', () => {
    // The recap end-card + its auto-advance must never interrupt live audio: crossing 95% mid-play
    // records the finish stat but must NOT raise the recap. Only a real stop (pause / ended) does.
    const el = stubAudio({ currentTime: 96, duration: 100 })
    const p = usePlayerStore()
    loaded(p, el)
    el.__emit('play') // playing = true
    p.onDurationChange()
    p.savePosition() // throttled flush during playback
    expect(p.justFinished).toBeNull()
    el.__emit('pause') // onPause flushes with playing=false → now the recap may raise
    expect(p.justFinished).toBe('ep-1')
  })

  it('justFinished stays null below the threshold', () => {
    const el = stubAudio({ currentTime: 50, duration: 100 })
    const p = usePlayerStore()
    loaded(p, el)
    p.onDurationChange()
    p.savePosition()
    expect(p.justFinished).toBeNull()
  })

  // --- advance hold: the recap end-card drives the advance, not onEnded (RFC-122 #2038) ---

  it('a held advance does NOT auto-play the next episode on `ended`', async () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.setAdvanceResolver(async () => ({ slug: 'ep-2', url: 'https://x/b.mp3', title: 'B' }))
    p.setAdvanceHold(true) // a mounted recap surface will drive it
    el.__emit('ended')
    await drainMicrotasks()
    // The finish is recorded, but the queue did NOT start — currentSlug is still the finished one.
    expect(p.justFinished).toBe('ep-1')
    expect(p.currentSlug).toBe('ep-1')
  })

  it('playNext() advances to the queued next when the end-card asks for it', async () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.setAdvanceResolver(async () => ({ slug: 'ep-2', url: 'https://x/b.mp3', title: 'B' }))
    p.setAdvanceHold(true)
    el.__emit('ended')
    await drainMicrotasks()
    const advanced = await p.playNext()
    await drainMicrotasks()
    expect(advanced).toBe(true)
    expect(p.currentSlug).toBe('ep-2')
  })

  it('unheld onEnded still auto-advances — audio outliving the view must not regress', async () => {
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    p.setAdvanceResolver(async () => ({ slug: 'ep-2', url: 'https://x/b.mp3', title: 'B' }))
    // no setAdvanceHold(true) → default false
    el.__emit('ended')
    await drainMicrotasks()
    expect(p.currentSlug).toBe('ep-2')
  })
})


describe('output routing — the system picker, never our own device list (operator 2026-09-23)', () => {
  /*
   * Two platform APIs of the same shape sit behind one control, and the store picks between them by
   * FEATURE, not by platform string — MDN marks Remote Playback "limited availability" and does not
   * say whether the Android System WebView carries it as opposed to Chrome.
   *
   * What is NOT tested, because it cannot be: the device list. Neither iOS nor Android exposes an
   * API for a page to enumerate AirPlay / Cast / Bluetooth targets, so the sheet is the platform's
   * and its contents are outside the app entirely.
   */
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => vi.unstubAllGlobals())

  it('starts with no route, so the control is hidden until a platform says otherwise', () => {
    const p = usePlayerStore()
    loaded(p, stubAudio())
    expect(p.routeAvailable).toBe(false)
    expect(p.playingRemotely).toBe(false)
  })

  it('iOS: the picker is always offered, AirPlay receivers or not (operator 2026-10-08)', () => {
    // It followed `webkitplaybacktargetavailabilitychanged`, which only reports AirPlay receivers;
    // away from one the button disappeared. The system sheet always has the iPhone and Bluetooth.
    const el = stubAudio({ webkitShowPlaybackTargetPicker: vi.fn() } as never)
    const p = usePlayerStore()
    loaded(p, el)
    expect(p.routeAvailable).toBe(true)
  })

  it('iOS: the picker call is delegated to the element, not reimplemented', () => {
    const show = vi.fn()
    const el = stubAudio({ webkitShowPlaybackTargetPicker: show } as never)
    const p = usePlayerStore()
    loaded(p, el)
    p.showRoutePicker()
    expect(show).toHaveBeenCalled()
  })

  it('Android: falls through to the Remote Playback API when webkit is absent', () => {
    const prompt = vi.fn(() => Promise.resolve())
    let availabilityCb: ((a: boolean) => void) | null = null
    const remote = {
      watchAvailability: vi.fn((cb: (a: boolean) => void) => {
        availabilityCb = cb
        return Promise.resolve(1)
      }),
      prompt,
      addEventListener: vi.fn(),
    }
    const el = stubAudio({ remote } as never)
    const p = usePlayerStore()
    loaded(p, el)

    expect(remote.watchAvailability).toHaveBeenCalled()
    availabilityCb!(true)
    expect(p.routeAvailable).toBe(true)

    p.showRoutePicker()
    expect(prompt).toHaveBeenCalled()
  })

  it('a platform with neither API is silent, not broken', () => {
    // A desktop browser, or an Android WebView without Remote Playback. The store must not throw —
    // these are optional APIs on a detached element, and a throw here costs the listener their
    // player over a control they cannot use anyway.
    const el = stubAudio()
    const p = usePlayerStore()
    loaded(p, el)
    expect(() => p.showRoutePicker()).not.toThrow()
    expect(p.routeAvailable).toBe(false)
  })
})


describe('cold-launch restore support (#2278)', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('nowPlaying is null with nothing loaded', () => {
    expect(usePlayerStore().nowPlaying()).toBeNull()
  })

  it('nowPlaying carries the ORIGIN url, not a local source, so the episode can be reloaded', () => {
    const p = usePlayerStore()
    p.setSourceResolver(() => 'capacitor://localhost/_capacitor_file_/ep.mp3')
    p.load({ slug: 'ep', url: 'https://cdn.example/ep.mp3', title: 'Ep', showTitle: 'Show' })
    expect(p.nowPlaying()).toMatchObject({
      slug: 'ep',
      url: 'https://cdn.example/ep.mp3',
      title: 'Ep',
      showTitle: 'Show',
      position: 0,
    })
    p.clear()
    expect(p.nowPlaying()).toBeNull()
  })

  it('loadAt waits for metadata before seeking — WebKit ignores an earlier currentTime — and never plays', () => {
    const p = usePlayerStore()
    p.loadAt({ slug: 'ep', url: 'https://cdn.example/ep.mp3' }, 754)
    const el = p.el!
    expect(p.currentTime).toBe(0)
    el.dispatchEvent(new Event('loadedmetadata'))
    expect(el.currentTime).toBe(754)
    expect(p.currentTime).toBe(754)
    expect(p.playing).toBe(false)
  })
})
