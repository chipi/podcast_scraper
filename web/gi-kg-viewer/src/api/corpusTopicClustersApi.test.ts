import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  fetchTopicClustersDocument,
  fetchTopicClustersFromApi,
  topicClustersSchemaWarning,
} from './corpusTopicClustersApi'

function expectFetchCalledWithUrl(expectedUrl: string): void {
  expect(fetch).toHaveBeenCalledWith(
    expectedUrl,
    expect.objectContaining({
      signal: expect.any(AbortSignal),
    }),
  )
}

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

describe('corpusTopicClustersApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  describe('fetchTopicClustersFromApi', () => {
    it('GETs /api/corpus/topic-clusters with trimmed path query', async () => {
      const payload = {
        schema_version: '2',
        clusters: [
          {
            graph_compound_parent_id: 'tc:g',
            cil_alias_target_topic_id: 'topic:a',
            canonical_label: 'A',
            members: [{ topic_id: 'topic:a' }],
          },
        ],
      }
      mockFetchJson(true, payload)
      await expect(fetchTopicClustersFromApi('  /corpus/root  ')).resolves.toEqual({
        status: 'ok',
        document: payload,
      })
      expectFetchCalledWithUrl('/api/corpus/topic-clusters?path=%2Fcorpus%2Froot')
    })

    it('omits query when corpus path is empty', async () => {
      mockFetchJson(true, { schema_version: '2', clusters: [] })
      await expect(fetchTopicClustersFromApi('')).resolves.toEqual({
        status: 'ok',
        document: { schema_version: '2', clusters: [] },
      })
      expectFetchCalledWithUrl('/api/corpus/topic-clusters')
    })

    it('returns missing on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(fetchTopicClustersFromApi('/x')).resolves.toEqual({ status: 'missing' })
    })

    it('returns error on other non-OK responses', async () => {
      mockFetchJson(false, {}, 'bad', 500)
      await expect(fetchTopicClustersFromApi('/x')).resolves.toEqual({
        status: 'error',
        message: 'bad',
      })
    })

    it('returns schemaWarning for unknown schema_version', async () => {
      mockFetchJson(true, { schema_version: '99', clusters: [] })
      await expect(fetchTopicClustersFromApi('/c')).resolves.toEqual({
        status: 'ok',
        document: { schema_version: '99', clusters: [] },
        schemaWarning: expect.stringContaining('Unknown topic_clusters schema_version'),
      })
    })
  })

  describe('fetchTopicClustersDocument (compat)', () => {
    it('returns document on ok', async () => {
      mockFetchJson(true, { schema_version: '2', clusters: [] })
      await expect(fetchTopicClustersDocument('/r')).resolves.toEqual({
        schema_version: '2',
        clusters: [],
      })
    })

    it('returns null on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(fetchTopicClustersDocument('/x')).resolves.toBeNull()
    })

    it('throws on other non-OK responses', async () => {
      mockFetchJson(false, {}, 'bad', 500)
      await expect(fetchTopicClustersDocument('/x')).rejects.toThrow('bad')
    })
  })

  describe('topicClustersSchemaWarning', () => {
    it('returns undefined for known versions', () => {
      expect(topicClustersSchemaWarning({ schema_version: '1' })).toBeUndefined()
      expect(topicClustersSchemaWarning({ schema_version: '2' })).toBeUndefined()
    })

    it('returns message for unknown version', () => {
      const w = topicClustersSchemaWarning({ schema_version: 'future' })
      expect(w).toContain('future')
      expect(w).toContain('supported')
    })

    it('returns undefined when schema_version absent', () => {
      expect(topicClustersSchemaWarning({})).toBeUndefined()
    })
  })
})
