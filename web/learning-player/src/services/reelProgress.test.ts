import { nextTick, ref } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({
  putReelProgress: vi.fn(async () => undefined),
  clearReelProgress: vi.fn(async () => undefined),
}))
vi.mock('./api', () => api)

import { watchReelProgress } from './reelProgress'
import type { ReelState } from '../stores/player'

const moments = [1, 2, 3].map((n) => ({ insightId: `i${n}`, text: '', speaker: null, startMs: 0, endMs: 1, slug: 'a' }))
const state = (index: number, done = false, id = 'thc:x'): ReelState => ({
  slug: 'a',
  topic: { id, label: 'X', minutes: 15 },
  moments,
  index,
  returnTo: 0,
  done,
})

function setup() {
  const reel = ref<ReelState | null>(null)
  const playing = ref(false)
  const left = ref<{ id: string; how: 'close' | 'keep' } | null>(null)
  const signedIn = ref(true)
  const stop = watchReelProgress({
    reel: () => reel.value,
    playing: () => playing.value,
    left: () => left.value,
    signedIn: () => signedIn.value,
  })
  return { reel, playing, left, signedIn, stop }
}

afterEach(() => vi.clearAllMocks())

describe('watchReelProgress', () => {
  it('writes nothing until something is heard; then each clip once; the end clears that reel', async () => {
    const t = setup()
    t.reel.value = state(0) // opened, nothing played (a blocked start, or just looked)
    await nextTick()
    expect(api.putReelProgress).not.toHaveBeenCalled()
    t.playing.value = true
    await nextTick()
    t.reel.value = { ...state(0) } // same clip: no second write
    await nextTick()
    t.reel.value = state(2)
    await nextTick()
    expect(api.putReelProgress.mock.calls).toEqual([
      ['thc:x', 0, 3, 15],
      ['thc:x', 2, 3, 15],
    ])
    t.reel.value = state(2, true)
    await nextTick()
    expect(api.clearReelProgress).toHaveBeenCalledWith('thc:x')
    t.stop()
  })

  it('moving past clip 1 counts as heard even if the sound never started', async () => {
    const t = setup()
    t.reel.value = state(1)
    await nextTick()
    expect(api.putReelProgress).toHaveBeenCalledWith('thc:x', 1, 3, 15)
    t.stop()
  })

  it('✕ forgets that reel; "Keep listening" keeps it to come back to', async () => {
    const t = setup()
    t.left.value = { id: 'thc:x', how: 'keep' }
    await nextTick()
    expect(api.clearReelProgress).not.toHaveBeenCalled()
    t.left.value = { id: 'thc:x', how: 'close' }
    await nextTick()
    expect(api.clearReelProgress).toHaveBeenCalledWith('thc:x')
    t.stop()
  })

  it("writes nothing signed out, and ignores an episode's own Moments reel (no topic)", async () => {
    const t = setup()
    t.signedIn.value = false
    t.reel.value = state(2)
    await nextTick()
    t.signedIn.value = true
    t.reel.value = { ...state(1), topic: undefined }
    await nextTick()
    expect(api.putReelProgress).not.toHaveBeenCalled()
    t.stop()
  })
})
