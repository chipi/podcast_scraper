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
    expect(toReel([M])).toEqual([{ insightId: 'i1', text: 'A point', speaker: 'Ann', startMs: 1000, endMs: 9000 }])
  })
})
