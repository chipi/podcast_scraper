import { afterEach, describe, expect, it, vi } from 'vitest'
import type { SearchResponse } from './searchApi'
import { searchCorpus } from './searchApi'

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

function calledUrl(): string {
  return vi.mocked(fetch).mock.calls[0][0] as string
}

function calledParams(): URLSearchParams {
  const url = calledUrl()
  return new URLSearchParams(url.slice(url.indexOf('?') + 1))
}

const emptyResponse: SearchResponse = { query: 'q', results: [] }

describe('searchApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  describe('searchCorpus query-param building', () => {
    it('GETs /api/search with trimmed path and q plus default top_k', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('  hello world  ', { path: '  /my/corpus  ' })
      const p = calledParams()
      expect(calledUrl()).toMatch(/^\/api\/search\?/)
      expect(p.get('path')).toBe('/my/corpus')
      expect(p.get('q')).toBe('hello world')
      expect(p.get('top_k')).toBe('10')
      // signal forwarded via fetchWithTimeout
      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      )
    })

    it('clamps topK to a minimum of 1', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', topK: 0 })
      expect(calledParams().get('top_k')).toBe('1')
    })

    it('clamps topK to a maximum of 100', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', topK: 5000 })
      expect(calledParams().get('top_k')).toBe('100')
    })

    it('passes a mid-range topK through unchanged', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', topK: 42 })
      expect(calledParams().get('top_k')).toBe('42')
    })

    it('clamps a negative topK up to 1', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', topK: -7 })
      expect(calledParams().get('top_k')).toBe('1')
    })

    it('sets grounded_only=true when groundedOnly is set', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', groundedOnly: true })
      expect(calledParams().get('grounded_only')).toBe('true')
    })

    it('omits grounded_only when groundedOnly is false', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', groundedOnly: false })
      expect(calledParams().has('grounded_only')).toBe(false)
    })

    it('sets trimmed feed/since/speaker/embedding_model when provided', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', {
        path: '/c',
        feed: '  feed-1  ',
        since: '  2024-01-01  ',
        speaker: '  Alice  ',
        embeddingModel: '  all-MiniLM  ',
      })
      const p = calledParams()
      expect(p.get('feed')).toBe('feed-1')
      expect(p.get('since')).toBe('2024-01-01')
      expect(p.get('speaker')).toBe('Alice')
      expect(p.get('embedding_model')).toBe('all-MiniLM')
    })

    it('omits feed/since/speaker/embedding_model when blank-only', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', {
        path: '/c',
        feed: '   ',
        since: '',
        speaker: '  ',
        embeddingModel: '   ',
      })
      const p = calledParams()
      expect(p.has('feed')).toBe(false)
      expect(p.has('since')).toBe(false)
      expect(p.has('speaker')).toBe(false)
      expect(p.has('embedding_model')).toBe(false)
    })

    it('appends each non-blank type and drops blank ones', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', {
        path: '/c',
        types: ['kg_entity', '  ', 'segment', ''],
      })
      const p = calledParams()
      expect(p.getAll('type')).toEqual(['kg_entity', 'segment'])
    })

    it('omits type entirely when types is undefined', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c' })
      expect(calledParams().has('type')).toBe(false)
    })

    it('sets dedupe_kg_surfaces=false only when dedupeKgSurfaces === false', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', dedupeKgSurfaces: false })
      expect(calledParams().get('dedupe_kg_surfaces')).toBe('false')
    })

    it('omits dedupe_kg_surfaces when dedupeKgSurfaces is true', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c', dedupeKgSurfaces: true })
      expect(calledParams().has('dedupe_kg_surfaces')).toBe(false)
    })

    it('omits dedupe_kg_surfaces when dedupeKgSurfaces is undefined', async () => {
      mockFetchJson(true, emptyResponse)
      await searchCorpus('q', { path: '/c' })
      expect(calledParams().has('dedupe_kg_surfaces')).toBe(false)
    })
  })

  describe('searchCorpus result parsing', () => {
    it('returns the parsed SearchResponse on success', async () => {
      const payload: SearchResponse = {
        query: 'climate',
        results: [
          {
            doc_id: 'd1',
            score: 0.9,
            metadata: { feed: 'f' },
            text: 'hello',
            source_tier: 'insight',
          },
        ],
        query_type: 'semantic',
        lift_stats: { transcript_hits_returned: 3, lift_applied: 1 },
        enrichment_error: null,
      }
      mockFetchJson(true, payload)
      await expect(searchCorpus('climate', { path: '/c' })).resolves.toEqual(payload)
    })

    it('returns an empty results array when the server has no hits', async () => {
      mockFetchJson(true, { query: 'q', results: [] })
      const r = await searchCorpus('q', { path: '/c' })
      expect(r.results).toEqual([])
    })
  })

  describe('searchCorpus error handling', () => {
    it('throws with response text when not ok', async () => {
      mockFetchJson(false, {}, 'index not built')
      await expect(searchCorpus('q', { path: '/c' })).rejects.toThrow('index not built')
    })

    it('throws HTTP status when not ok and body is empty', async () => {
      mockFetchJson(false, {}, '', 503)
      await expect(searchCorpus('q', { path: '/c' })).rejects.toThrow('HTTP 503')
    })

    it('propagates a network/fetch rejection', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => {
          throw new Error('network down')
        }) as unknown as typeof fetch,
      )
      await expect(searchCorpus('q', { path: '/c' })).rejects.toThrow('network down')
    })

    it('propagates a malformed-JSON parse error on an ok response', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          ok: true,
          status: 200,
          text: async () => '',
          json: async () => {
            throw new SyntaxError('Unexpected token < in JSON')
          },
        })) as unknown as typeof fetch,
      )
      await expect(searchCorpus('q', { path: '/c' })).rejects.toThrow(/Unexpected token/)
    })
  })
})
