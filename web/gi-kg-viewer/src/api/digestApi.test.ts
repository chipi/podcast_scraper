import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchCorpusDigest } from './digestApi'

function mockFetchJson(
  ok: boolean,
  body: unknown,
  text = '',
  status?: number,
): void {
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

describe('digestApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('GETs /api/corpus/digest with trimmed path and default query', async () => {
    const payload = {
      path: '/r',
      window: '7d' as const,
      window_start_utc: 'a',
      window_end_utc: 'b',
      compact: false,
      rows: [],
      topics: [],
      topics_unavailable_reason: null,
    }
    mockFetchJson(true, payload)
    await expect(fetchCorpusDigest('  /c  ')).resolves.toEqual(payload)
    expect(fetch).toHaveBeenCalledWith('/api/corpus/digest?path=%2Fc')
  })

  it('passes window, since, compact, include_topics, max_rows when set', async () => {
    const payload = {
      path: '/r',
      window: 'since' as const,
      window_start_utc: 'a',
      window_end_utc: 'b',
      compact: true,
      rows: [],
      topics: [],
      topics_unavailable_reason: null,
    }
    mockFetchJson(true, payload)
    await fetchCorpusDigest('/c', {
      window: 'since',
      since: '2024-01-15',
      compact: true,
      includeTopics: false,
      maxRows: 12,
    })
    expect(fetch).toHaveBeenCalledWith(
      '/api/corpus/digest?path=%2Fc&window=since&since=2024-01-15&compact=true&include_topics=false&max_rows=12',
    )
  })

  it('throws upgrade hint on 404', async () => {
    mockFetchJson(false, {}, '', 404)
    await expect(fetchCorpusDigest('/x')).rejects.toThrow(/Digest endpoint not found/)
  })

  it('throws response text when not ok', async () => {
    mockFetchJson(false, {}, 'bad digest')
    await expect(fetchCorpusDigest('/x')).rejects.toThrow('bad digest')
  })
})
