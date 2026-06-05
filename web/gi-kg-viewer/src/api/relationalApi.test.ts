import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  fetchCrossShow,
  fetchEpisodeRelatedInsights,
  fetchPositions,
  fetchShowEpisodes,
  fetchWhoSaid,
} from './relationalApi'

function expectFetchUrl(expectedUrl: string): void {
  expect(fetch).toHaveBeenCalledWith(
    expectedUrl,
    expect.objectContaining({ signal: expect.any(AbortSignal) }),
  )
}

function mockFetchJson(ok: boolean, body: unknown, text = '', status?: number): void {
  const st = status ?? (ok ? 200 : 500)
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      status: st,
      text: async () => text,
      json: async () => body,
    })) as unknown as typeof fetch,
  )
}

describe('relationalApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('fetchCrossShow GETs cross-show with trimmed path + topic', async () => {
    const payload = { subject: 'topic:ai', groups: { 'podcast:s1': [] }, error: null }
    mockFetchJson(true, payload)
    await expect(fetchCrossShow('  /c  ', '  topic:ai  ')).resolves.toEqual(payload)
    expectFetchUrl('/api/relational/cross-show?path=%2Fc&topic=topic%3Aai')
  })

  it('fetchCrossShow adds per_show only when not 1', async () => {
    mockFetchJson(true, { subject: 'topic:ai', groups: {} })
    await fetchCrossShow('/c', 'topic:ai', 3)
    expectFetchUrl('/api/relational/cross-show?path=%2Fc&topic=topic%3Aai&per_show=3')
  })

  it('fetchWhoSaid sets k when provided', async () => {
    mockFetchJson(true, { subject: 'topic:ai', groups: {} })
    await fetchWhoSaid('/c', 'topic:ai', 5)
    expectFetchUrl('/api/relational/who-said?path=%2Fc&topic=topic%3Aai&k=5')
  })

  it('fetchPositions GETs the person positions list', async () => {
    const payload = { subject: 'person:x', results: [], error: null }
    mockFetchJson(true, payload)
    await expect(fetchPositions('/c', 'person:x')).resolves.toEqual(payload)
    expectFetchUrl('/api/relational/positions?path=%2Fc&person=person%3Ax')
  })

  it('fetchShowEpisodes GETs the podcast episodes list', async () => {
    mockFetchJson(true, { subject: 'podcast:s1', results: [] })
    await fetchShowEpisodes('/c', 'podcast:s1')
    expectFetchUrl('/api/relational/episodes?path=%2Fc&podcast=podcast%3As1')
  })

  it('fetchEpisodeRelatedInsights GETs episode-insights with k', async () => {
    mockFetchJson(true, { subject: 'e1', results: [] })
    await fetchEpisodeRelatedInsights('/c', 'e1', 10)
    expectFetchUrl('/api/relational/episode-insights?path=%2Fc&episode=e1&k=10')
  })

  it('raises a friendly error on 404', async () => {
    mockFetchJson(false, {}, 'nope', 404)
    await expect(fetchCrossShow('/c', 'topic:ai')).rejects.toThrow(/not found \(404\)/)
  })

  it('propagates body text on other errors', async () => {
    mockFetchJson(false, {}, 'boom', 500)
    await expect(fetchWhoSaid('/c', 'topic:ai')).rejects.toThrow('boom')
  })
})
