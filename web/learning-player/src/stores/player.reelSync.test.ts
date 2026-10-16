import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const clipWindow = vi.fn()
vi.mock('../player/audioSyncRegistry', () => ({ clipWindow: (...a: unknown[]) => clipWindow(...a) }))

import { usePlayerStore } from './player'

/** RFC-129: before a reel clip plays, it is re-measured on this listener's own stream. */

function stubAudio() {
  const el = {
    paused: true,
    currentTime: 1234,
    duration: 1800,
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
    addEventListener: vi.fn(),
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

const CLIP = {
  insightId: 'i1',
  text: 'one',
  speaker: 'Ann',
  // Where the server measured it, and where it is in the transcript.
  startMs: 70_000,
  endMs: 85_000,
  transcriptStartMs: 60_000,
  transcriptEndMs: 75_000,
}

describe('reel clips on the listener stream (#2359)', () => {
  let el: ReturnType<typeof stubAudio>
  let p: ReturnType<typeof usePlayerStore>
  beforeEach(() => {
    clipWindow.mockReset()
    setActivePinia(createPinia())
    p = usePlayerStore()
    el = stubAudio()
    p.load({ slug: 'ep-1', url: 'https://x/a.mp3', title: 'An Episode' })
    p.onDurationChange()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('plays the clip where this listener stream has it', async () => {
    clipWindow.mockResolvedValue({ startMs: 95_000, endMs: 110_000 })
    expect(p.startReel('ep-1', [CLIP])).toBe(true)
    await drain()
    expect(clipWindow).toHaveBeenCalledWith('ep-1', 'https://x/a.mp3', 60_000, 75_000)
    expect(el.currentTime).toBe(95)
    expect(p.reel?.moments[0]).toMatchObject({ startMs: 95_000, endMs: 110_000 })
  })

  it('keeps the server window when the stream cannot be measured', async () => {
    clipWindow.mockResolvedValue(null)
    p.startReel('ep-1', [CLIP])
    await drain()
    expect(el.currentTime).toBe(70)
  })

  it('does not measure a clip without transcript times', async () => {
    p.startReel('ep-1', [{ insightId: 'i1', text: 'one', speaker: null, startMs: 70_000, endMs: 85_000 }])
    await drain()
    expect(clipWindow).not.toHaveBeenCalled()
    expect(el.currentTime).toBe(70)
  })
})
