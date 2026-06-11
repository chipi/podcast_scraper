import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchExploreFiltered, fetchExploreNaturalLanguage } from './exploreApi'

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

function lastUrl(): string {
  const calls = vi.mocked(fetch).mock.calls
  return String(calls[calls.length - 1]?.[0])
}

function lastParams(): URLSearchParams {
  const url = lastUrl()
  return new URLSearchParams(url.slice(url.indexOf('?') + 1))
}

describe('exploreApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  describe('fetchExploreFiltered', () => {
    it('GETs /api/explore with trimmed path and sensible defaults', async () => {
      const payload = { kind: 'explore' as const, data: { items: [] } }
      mockFetchJson(true, payload)
      await expect(fetchExploreFiltered('  /corpus  ', {})).resolves.toEqual(payload)
      const params = lastParams()
      expect(lastUrl()).toContain('/api/explore?')
      expect(params.get('path')).toBe('/corpus')
      // defaults
      expect(params.get('sort_by')).toBe('confidence')
      expect(params.get('limit')).toBe('50')
      // optional params unset
      expect(params.has('topic')).toBe(false)
      expect(params.has('speaker')).toBe(false)
      expect(params.has('grounded_only')).toBe(false)
      expect(params.has('min_confidence')).toBe(false)
      expect(params.has('strict')).toBe(false)
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    })

    it('forwards trimmed topic and speaker when non-empty', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { topic: '  ai  ', speaker: '  bob  ' })
      const params = lastParams()
      expect(params.get('topic')).toBe('ai')
      expect(params.get('speaker')).toBe('bob')
    })

    it('omits topic/speaker when only whitespace', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { topic: '   ', speaker: '' })
      const params = lastParams()
      expect(params.has('topic')).toBe(false)
      expect(params.has('speaker')).toBe(false)
    })

    it('sets grounded_only only when truthy', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { groundedOnly: true })
      expect(lastParams().get('grounded_only')).toBe('true')

      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { groundedOnly: false })
      expect(lastParams().has('grounded_only')).toBe(false)
    })

    it('sets min_confidence when a finite number', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { minConfidence: 0.75 })
      expect(lastParams().get('min_confidence')).toBe('0.75')
    })

    it('passes min_confidence of 0 (finite, non-null)', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { minConfidence: 0 })
      expect(lastParams().get('min_confidence')).toBe('0')
    })

    it('omits min_confidence when null', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { minConfidence: null })
      expect(lastParams().has('min_confidence')).toBe(false)
    })

    it('omits min_confidence when undefined', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', {})
      expect(lastParams().has('min_confidence')).toBe(false)
    })

    it('omits min_confidence when non-finite (NaN/Infinity)', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { minConfidence: Number.NaN })
      expect(lastParams().has('min_confidence')).toBe(false)

      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { minConfidence: Number.POSITIVE_INFINITY })
      expect(lastParams().has('min_confidence')).toBe(false)
    })

    it('honours an explicit sortBy of "time"', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { sortBy: 'time' })
      expect(lastParams().get('sort_by')).toBe('time')
    })

    it('clamps limit into [1, 500]', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { limit: 9999 })
      expect(lastParams().get('limit')).toBe('500')

      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { limit: 0 })
      expect(lastParams().get('limit')).toBe('1')

      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { limit: -10 })
      expect(lastParams().get('limit')).toBe('1')

      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { limit: 42 })
      expect(lastParams().get('limit')).toBe('42')
    })

    it('sets strict only when truthy', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { strict: true })
      expect(lastParams().get('strict')).toBe('true')

      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', { strict: false })
      expect(lastParams().has('strict')).toBe(false)
    })

    it('passes an AbortSignal via fetchWithTimeout', async () => {
      mockFetchJson(true, { kind: 'explore' })
      await fetchExploreFiltered('/c', {})
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/explore?'),
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      )
    })

    it('throws the response text on a non-ok status', async () => {
      mockFetchJson(false, {}, 'boom explore', 400)
      await expect(fetchExploreFiltered('/c', {})).rejects.toThrow('boom explore')
    })

    it('throws "HTTP <status>" when error body is empty', async () => {
      mockFetchJson(false, {}, '', 503)
      await expect(fetchExploreFiltered('/c', {})).rejects.toThrow('HTTP 503')
    })

    it('propagates a network error from fetch', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => {
          throw new Error('network down')
        }) as unknown as typeof fetch,
      )
      await expect(fetchExploreFiltered('/c', {})).rejects.toThrow('network down')
    })

    it('propagates malformed JSON on a 200 response', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          ok: true,
          status: 200,
          text: async () => '',
          json: async () => {
            throw new SyntaxError('Unexpected token')
          },
        })) as unknown as typeof fetch,
      )
      await expect(fetchExploreFiltered('/c', {})).rejects.toThrow(/Unexpected token/)
    })
  })

  describe('fetchExploreNaturalLanguage', () => {
    it('GETs /api/explore with trimmed q, path and default limit', async () => {
      const payload = {
        kind: 'natural_language' as const,
        question: 'who said x',
        answer: { text: 'y' },
      }
      mockFetchJson(true, payload)
      await expect(
        fetchExploreNaturalLanguage('  /corpus  ', '  who said x  '),
      ).resolves.toEqual(payload)
      const params = lastParams()
      expect(lastUrl()).toContain('/api/explore?')
      expect(params.get('path')).toBe('/corpus')
      expect(params.get('q')).toBe('who said x')
      expect(params.get('limit')).toBe('50')
      expect(params.has('strict')).toBe(false)
    })

    it('clamps limit into [1, 500]', async () => {
      mockFetchJson(true, { kind: 'natural_language' })
      await fetchExploreNaturalLanguage('/c', 'q', { limit: 9999 })
      expect(lastParams().get('limit')).toBe('500')

      mockFetchJson(true, { kind: 'natural_language' })
      await fetchExploreNaturalLanguage('/c', 'q', { limit: 0 })
      expect(lastParams().get('limit')).toBe('1')
    })

    it('sets strict only when truthy', async () => {
      mockFetchJson(true, { kind: 'natural_language' })
      await fetchExploreNaturalLanguage('/c', 'q', { strict: true })
      expect(lastParams().get('strict')).toBe('true')

      mockFetchJson(true, { kind: 'natural_language' })
      await fetchExploreNaturalLanguage('/c', 'q', { strict: false })
      expect(lastParams().has('strict')).toBe(false)
    })

    it('defaults options to {} when omitted', async () => {
      mockFetchJson(true, { kind: 'natural_language' })
      await fetchExploreNaturalLanguage('/c', 'q')
      const params = lastParams()
      expect(params.get('limit')).toBe('50')
      expect(params.has('strict')).toBe(false)
    })

    it('throws the response text on a non-ok status', async () => {
      mockFetchJson(false, {}, 'boom nl', 400)
      await expect(fetchExploreNaturalLanguage('/c', 'q')).rejects.toThrow('boom nl')
    })

    it('throws "HTTP <status>" when error body is empty', async () => {
      mockFetchJson(false, {}, '', 500)
      await expect(fetchExploreNaturalLanguage('/c', 'q')).rejects.toThrow('HTTP 500')
    })

    it('propagates a network error from fetch', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => {
          throw new Error('network down')
        }) as unknown as typeof fetch,
      )
      await expect(fetchExploreNaturalLanguage('/c', 'q')).rejects.toThrow('network down')
    })
  })
})
