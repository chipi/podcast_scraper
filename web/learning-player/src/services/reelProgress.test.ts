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
const state = (index: number, done = false): ReelState => ({
  slug: 'a',
  topic: { id: 'thc:x', label: 'X', minutes: 15 },
  moments,
  index,
  returnTo: 0,
  done,
})

afterEach(() => vi.clearAllMocks())

describe('watchReelProgress', () => {
  it('writes each clip of a short listen once, clears it at the end, and nothing signed out', async () => {
    const reel = ref<ReelState | null>(null)
    const signedIn = ref(true)
    const stop = watchReelProgress(() => reel.value, () => signedIn.value)
    reel.value = state(0)
    await nextTick()
    reel.value = { ...state(0) } // same clip: no second write
    await nextTick()
    reel.value = state(2)
    await nextTick()
    expect(api.putReelProgress.mock.calls).toEqual([
      ['thc:x', 0, 3],
      ['thc:x', 2, 3],
    ])
    reel.value = state(2, true)
    await nextTick()
    expect(api.clearReelProgress).toHaveBeenCalledOnce()
    signedIn.value = false
    reel.value = state(1)
    await nextTick()
    expect(api.putReelProgress).toHaveBeenCalledTimes(2)
    stop()
  })

  it("ignores an episode's own Moments reel (no topic)", async () => {
    const reel = ref<ReelState | null>(null)
    const stop = watchReelProgress(() => reel.value, () => true)
    reel.value = { ...state(1), topic: undefined }
    await nextTick()
    expect(api.putReelProgress).not.toHaveBeenCalled()
    stop()
  })
})
