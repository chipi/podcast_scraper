import { beforeEach, describe, expect, it, vi } from 'vitest'

const getAudioReference = vi.fn()
vi.mock('../services/api', () => ({ getAudioReference: (...a: unknown[]) => getAudioReference(...a) }))
const localKnowledgeFor = vi.fn()
vi.mock('../services/downloads', () => ({ localKnowledgeFor: (...a: unknown[]) => localKnowledgeFor(...a) }))
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
    localKnowledgeFor.mockReset()
    localKnowledgeFor.mockResolvedValue(null)
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

  it('offline, a downloaded episode syncs from the reference stored with it', async () => {
    getAudioReference.mockRejectedValue(new Error('offline'))
    localKnowledgeFor.mockResolvedValue({ audioReference: { scale: 20, window_s: 10, duration_s: 600, checkpoints: [{ t: 0, kind: 'talk_start', frames: '' }] } })
    ensure.mockResolvedValue([[0, 5]])
    expect(await syncFor('ep', 'capacitor://localhost/_capacitor_file_/ep.mp3')).not.toBeNull()
    expect(await clipWindow('ep', 'capacitor://localhost/_capacitor_file_/ep.mp3', 10_000, 20_000)).toEqual({ startMs: 15_000, endMs: 25_000 })
  })
})
