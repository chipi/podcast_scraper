import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from './api'
import * as downloads from './downloads'
import { loadMoments, toReel } from './moments'

const M = { insight_id: 'i1', text: 'A point', speaker: 'Ann', start_ms: 1000, end_ms: 9000, clip_text: 'x' }

describe('loadMoments (operator 2026-10-10)', () => {
  beforeEach(() => vi.restoreAllMocks())

  it('online, the server answers', async () => {
    vi.spyOn(api, 'getMoments').mockResolvedValue({ episode_slug: 'e', moments: [M], total_seconds: 8 })
    const local = vi.spyOn(downloads, 'localKnowledgeFor')
    expect(await loadMoments('e')).toEqual([M])
    expect(local).not.toHaveBeenCalled()
  })

  it('offline, the copy saved with the download', async () => {
    vi.spyOn(api, 'getMoments').mockRejectedValue(new api.ApiError(0, 'offline'))
    vi.spyOn(downloads, 'localKnowledgeFor').mockResolvedValue({
      detail: null, insights: [], topics: [], persons: [], moments: [M],
    })
    expect(await loadMoments('e')).toEqual([M])
  })

  it('neither: an empty reel, not an error', async () => {
    vi.spyOn(api, 'getMoments').mockRejectedValue(new Error('down'))
    vi.spyOn(downloads, 'localKnowledgeFor').mockResolvedValue(null)
    expect(await loadMoments('e')).toEqual([])
  })

  it('maps to the player store shape', () => {
    expect(toReel([M])).toEqual([
      {
        insightId: 'i1',
        text: 'A point',
        speaker: 'Ann',
        startMs: 1000,
        endMs: 9000,
        transcriptStartMs: 1000,
        transcriptEndMs: 9000,
        translatedFrom: null,
      },
    ])
  })

  it('plays at the server window, and keeps the transcript times to re-measure (#2359)', () => {
    const [r] = toReel([{ ...M, play_start_ms: 13000, play_end_ms: 21000 }])
    expect([r.startMs, r.endMs, r.transcriptStartMs, r.transcriptEndMs]).toEqual([13000, 21000, 1000, 9000])
  })

  it('carries a translated moment\'s source language (D-36 b)', () => {
    expect(toReel([{ ...M, translated_from: 'es' }])[0].translatedFrom).toBe('es')
  })
})
