import { beforeEach, describe, expect, it, vi } from 'vitest'

const getAudioReference = vi.fn()
vi.mock('../services/api', () => ({ getAudioReference: (...a: unknown[]) => getAudioReference(...a) }))
const ensure = vi.fn()
vi.mock('./audioSync', () => ({
  AudioSync: vi.fn(function (this: { ensure: unknown }) {
    this.ensure = ensure
  }),
  browserStreamAccess: vi.fn(),
}))

import { _resetForTests, clipWindow, syncFor } from './audioSyncRegistry'

describe('clip windows on the listener stream (RFC-129)', () => {
  beforeEach(() => {
    _resetForTests()
    getAudioReference.mockReset()
    ensure.mockReset()
  })

  it('converts the clip through the map measured on this stream, keeping its length', async () => {
    getAudioReference.mockResolvedValue({ scale: 20, window_s: 10, duration_s: 600, checkpoints: [{ t: 0, kind: 'talk_start', frames: '' }] })
    ensure.mockResolvedValue([
      [0, 12],
      [300, -40],
    ])
    expect(await clipWindow('ep', 'u', 100_000, 115_000)).toEqual({ startMs: 112_000, endMs: 127_000 })
    expect(await clipWindow('ep', 'u', 400_000, 410_000)).toEqual({ startMs: 360_000, endMs: 370_000 })
    expect(getAudioReference).toHaveBeenCalledTimes(1) // one sync per episode and stream
  })

  it('has nothing to say for an episode without a reference', async () => {
    getAudioReference.mockResolvedValue(null)
    expect(await syncFor('ep', 'u')).toBeNull()
    expect(await clipWindow('ep', 'u', 100_000, 115_000)).toBeNull()
  })
})
