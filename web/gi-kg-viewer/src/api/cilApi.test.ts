import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  fetchPersonProfile,
  fetchTopicPersons,
  fetchTopicTimeline,
  fetchTopicTimelineMerged,
} from './cilApi'

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

  it('fetchTopicTimeline throws when topic id empty', async () => {
    await expect(fetchTopicTimeline('/c', '   ')).rejects.toThrow(/Topic id/)
  })

  it('fetchTopicTimeline propagates trimmed body text on non-OK', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        text: async () => '  boom detail  ',
      })) as unknown as typeof fetch,
    )
    await expect(fetchTopicTimeline('/c', 'topic:t')).rejects.toThrow('boom detail')
  })

  it('fetchTopicTimeline falls back to HTTP status when body empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 503,
        text: async () => '   ',
      })) as unknown as typeof fetch,
    )
    await expect(fetchTopicTimeline('/c', 'topic:t')).rejects.toThrow('HTTP 503')
  })

  it('fetchTopicTimelineMerged throws when corpus path empty', async () => {
    await expect(fetchTopicTimelineMerged('   ', ['topic:a'])).rejects.toThrow(/Corpus path/)
  })

  it('fetchTopicTimelineMerged throws when no usable topic ids', async () => {
    await expect(fetchTopicTimelineMerged('/c', ['  ', ''])).rejects.toThrow(
      /At least one topic id/,
    )
  })

  it('fetchTopicTimelineMerged trims/filters ids and sets insight_types', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ path: '/c', topic_ids: ['topic:a'], episodes: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await fetchTopicTimelineMerged('/c', ['  topic:a  ', '', '  '], { insightTypes: '  claim  ' })
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(String(init?.body))
    expect(body.topic_ids).toEqual(['topic:a'])
    expect(body.insight_types).toBe('claim')
    fetchSpy.mockRestore()
  })

  it('fetchTopicTimelineMerged omits insight_types when blank', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ path: '/c', topic_ids: ['topic:a'], episodes: [] }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await fetchTopicTimelineMerged('/c', ['topic:a'], { insightTypes: '   ' })
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(String(init?.body))
    expect(body).not.toHaveProperty('insight_types')
    fetchSpy.mockRestore()
  })

  it('fetchTopicTimelineMerged propagates body text on non-OK', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 502,
        text: async () => 'merged boom',
      })) as unknown as typeof fetch,
    )
    await expect(fetchTopicTimelineMerged('/c', ['topic:a'])).rejects.toThrow('merged boom')
  })

  it('fetchTopicTimelineMerged falls back to HTTP status when body empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        text: async () => '',
      })) as unknown as typeof fetch,
    )
    await expect(fetchTopicTimelineMerged('/c', ['topic:a'])).rejects.toThrow('HTTP 500')
  })

  it('fetchTopicPersons throws when corpus path empty', async () => {
    await expect(fetchTopicPersons('   ', 'topic:x')).rejects.toThrow(/Corpus path/)
  })

  it('fetchTopicPersons throws when topic id empty', async () => {
    await expect(fetchTopicPersons('/c', '   ')).rejects.toThrow(/Topic id/)
  })

  it('fetchTopicPersons propagates body text on non-OK', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        text: async () => 'persons boom',
      })) as unknown as typeof fetch,
    )
    await expect(fetchTopicPersons('/root', 'topic:x')).rejects.toThrow('persons boom')
  })

  it('fetchTopicPersons falls back to HTTP status when body empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 404,
        text: async () => '',
      })) as unknown as typeof fetch,
    )
    await expect(fetchTopicPersons('/root', 'topic:x')).rejects.toThrow('HTTP 404')
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

  // #909 — corpus-wide person profile (/api/persons/{id}/brief)
  it('fetchPersonProfile GETs encoded person id + path and parses quotes', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url: string) => {
        expect(url).toContain('/api/persons/')
        expect(url).toContain(encodeURIComponent('person:satya-nadella'))
        expect(url).toContain('/brief?')
        expect(url).toContain('path=%2Fcorpus')
        return {
          ok: true,
          json: async () => ({
            path: '/corpus',
            person_id: 'person:satya-nadella',
            topics: {},
            quotes: [
              { episode_id: 'ep1', quote: { id: 'q1', properties: { text: 'hello' } } },
              { episode_id: 'ep2', quote: { id: 'q2', properties: { text: 'world' } } },
            ],
          }),
        }
      }) as unknown as typeof fetch,
    )
    const body = await fetchPersonProfile('/corpus', 'person:satya-nadella')
    expect(body.quotes).toHaveLength(2)
    expect(body.quotes[0].episode_id).toBe('ep1')
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
  })

  it('fetchPersonProfile throws when corpus path empty', async () => {
    await expect(fetchPersonProfile('   ', 'person:x')).rejects.toThrow(/Corpus path/)
  })

  it('fetchPersonProfile throws when person id empty', async () => {
    await expect(fetchPersonProfile('/c', '   ')).rejects.toThrow(/Person id/)
  })

  it('fetchPersonProfile propagates body text on non-OK', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 500,
        text: async () => '  profile boom  ',
      })) as unknown as typeof fetch,
    )
    await expect(fetchPersonProfile('/c', 'person:x')).rejects.toThrow('profile boom')
  })

  it('fetchPersonProfile falls back to HTTP status when body empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 404,
        text: async () => '',
      })) as unknown as typeof fetch,
    )
    await expect(fetchPersonProfile('/c', 'person:x')).rejects.toThrow('HTTP 404')
  })
})
