import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchTopicPersons, fetchTopicTimeline, fetchTopicTimelineMerged } from './cilApi'

describe('cilApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('fetchTopicTimeline encodes merged-graph g:topic ids', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        expect(url).toContain(encodeURIComponent('g:topic:tax'))
        return {
          ok: true,
          json: async () => ({
            path: '/corpus',
            topic_id: 'topic:tax',
            episodes: [],
          }),
        }
      }) as unknown as typeof fetch,
    )
    await fetchTopicTimeline('/corpus', 'g:topic:tax')
  })

  it('fetchTopicTimeline GETs encoded topic id and path', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        expect(url).toContain('/api/topics/')
        expect(url).toContain(encodeURIComponent('topic:tax'))
        expect(url).toContain('path=%2Fcorpus')
        return {
          ok: true,
          json: async () => ({
            path: '/corpus',
            topic_id: 'topic:tax',
            episodes: [],
          }),
        }
      }) as unknown as typeof fetch,
    )
    const body = await fetchTopicTimeline('/corpus', 'topic:tax')
    expect(body.episodes).toEqual([])
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
  })

  it('fetchTopicTimeline passes insight_types when set', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        expect(url).toContain('insight_types=claim')
        return {
          ok: true,
          json: async () => ({
            path: '/c',
            topic_id: 'topic:t',
            episodes: [],
          }),
        }
      }) as unknown as typeof fetch,
    )
    await fetchTopicTimeline('/c', 'topic:t', { insightTypes: 'claim' })
  })

  it('fetchTopicTimelineMerged POSTs topic_ids and path', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ path: '/c', topic_ids: ['topic:a'], episodes: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await fetchTopicTimelineMerged('/c', ['topic:a', 'topic:b'])
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/topics/timeline')
    expect(init?.method).toBe('POST')
    const body = JSON.parse(String(init?.body))
    expect(body.topic_ids).toEqual(['topic:a', 'topic:b'])
    expect(body.path).toBe('/c')
    fetchSpy.mockRestore()
  })

  it('fetchTopicTimeline throws when corpus path empty', async () => {
    await expect(fetchTopicTimeline('  ', 'topic:x')).rejects.toThrow(/Corpus path/)
  })

  it('fetchTopicPersons GETs encoded path', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        expect(url).toContain('/persons?')
        expect(url).toMatch(/\/api\/topics\/[^/]+\/persons\?/)
        expect(url).toContain('path=%2Froot')
        return {
          ok: true,
          json: async () => ({
            path: '/root',
            anchor_id: 'topic:x',
            ids: ['person:a'],
          }),
        }
      }) as unknown as typeof fetch,
    )
    const body = await fetchTopicPersons('/root', 'topic:x')
    expect(body.ids).toEqual(['person:a'])
  })
})
