import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CorpusSimilarEpisodesResponse } from './corpusLibraryApi'
import {
  fetchCorpusEpisodeDetail,
  fetchCorpusEpisodes,
  fetchCorpusFeeds,
  fetchCorpusSimilarEpisodes,
  fetchNodeEpisodes,
  fetchResolveEpisodeArtifacts,
  GRAPH_NODE_EPISODES_EXPAND_MAX,
} from './corpusLibraryApi'

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

describe('corpusLibraryApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  describe('fetchCorpusFeeds', () => {
    it('GETs /api/corpus/feeds with trimmed path query', async () => {
      const payload = { path: '/resolved', feeds: [] }
      mockFetchJson(true, payload)
      await expect(fetchCorpusFeeds('  /my/corpus  ')).resolves.toEqual(payload)
      expectFetchCalledWithUrl('/api/corpus/feeds?path=%2Fmy%2Fcorpus')
    })

    it('throws with response text when not ok', async () => {
      mockFetchJson(false, {}, 'missing corpus')
      await expect(fetchCorpusFeeds('/x')).rejects.toThrow('missing corpus')
    })

    it('throws HTTP status when not ok and empty body', async () => {
      mockFetchJson(false, {}, '')
      await expect(fetchCorpusFeeds('/x')).rejects.toThrow('HTTP 500')
    })

    it('throws upgrade hint on 404', async () => {
      mockFetchJson(false, {}, '{"detail":"Not Found"}', 404)
      await expect(fetchCorpusFeeds('/x')).rejects.toThrow(/Corpus Library endpoint not found/)
    })

    it('dedupes concurrent fetchCorpusFeeds for the same path into one HTTP call', async () => {
      const payload = { path: '/c', feeds: [] }
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => ({
          ok: true,
          status: 200,
          text: async () => '',
          json: async () => payload,
        })) as unknown as typeof fetch,
      )
      const [a, b] = await Promise.all([fetchCorpusFeeds('/c'), fetchCorpusFeeds('/c')])
      expect(a).toEqual(payload)
      expect(b).toEqual(payload)
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    })
  })

  describe('fetchCorpusEpisodes', () => {
    it('omits feed_id when options.feedId is undefined', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', {})
      expectFetchCalledWithUrl('/api/corpus/episodes?path=%2Fc')
    })

    it('includes feed_id for empty string (ungrouped filter)', async () => {
      const payload = { path: '/r', feed_id: '', items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { feedId: '' })
      expectFetchCalledWithUrl('/api/corpus/episodes?path=%2Fc&feed_id=')
    })

    it('passes q, since, limit, cursor when set', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', {
        feedId: 'f1',
        q: '  hello  ',
        since: ' 2024-01-01 ',
        limit: 25,
        cursor: 'abc',
      })
      const called = vi.mocked(fetch).mock.calls[0][0] as string
      expect(called).toContain('path=%2Fc')
      expect(called).toContain('feed_id=f1')
      expect(called).toContain('q=hello')
      expect(called).toContain('since=2024-01-01')
      expect(called).toContain('limit=25')
      expect(called).toContain('cursor=abc')
    })

    it('skips q and since when blank', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { q: '   ', since: '' })
      const called = vi.mocked(fetch).mock.calls[0][0] as string
      expect(called).not.toMatch(/[&?]q=/)
      expect(called).not.toMatch(/[&?]since=/)
    })

    it('passes topic_q when topicQ is set', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { topicQ: '  climate  ' })
      const called = vi.mocked(fetch).mock.calls[0][0] as string
      expect(called).toContain('topic_q=climate')
    })

    it('passes topic_cluster_only when topicClusterOnly is true', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { topicClusterOnly: true })
      const called = vi.mocked(fetch).mock.calls[0][0] as string
      expect(called).toContain('topic_cluster_only=true')
    })

    it('passes until when set and skips it when blank', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { until: '  2025-12-31  ' })
      expect(vi.mocked(fetch).mock.calls[0][0] as string).toContain('until=2025-12-31')

      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { until: '   ' })
      expect(vi.mocked(fetch).mock.calls[0][0] as string).not.toMatch(/[&?]until=/)
    })

    it('sets has_gi=true when hasGi is true', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { hasGi: true })
      expect(vi.mocked(fetch).mock.calls[0][0] as string).toContain('has_gi=true')
    })

    it('sets has_gi=false when hasGi is false', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { hasGi: false })
      expect(vi.mocked(fetch).mock.calls[0][0] as string).toContain('has_gi=false')
    })

    it('omits has_gi when hasGi is undefined', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', {})
      expect(vi.mocked(fetch).mock.calls[0][0] as string).not.toMatch(/has_gi=/)
    })

    it('includes limit=0 (limit != null) and omits a null cursor', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { limit: 0, cursor: null })
      const called = vi.mocked(fetch).mock.calls[0][0] as string
      expect(called).toContain('limit=0')
      expect(called).not.toMatch(/cursor=/)
    })

    it('returns total when the server supplies it', async () => {
      const payload = {
        path: '/r',
        feed_id: null,
        items: [],
        next_cursor: 'next-page',
        total: 42,
      }
      mockFetchJson(true, payload)
      const r = await fetchCorpusEpisodes('/c', {})
      expect(r.total).toBe(42)
      expect(r.next_cursor).toBe('next-page')
    })

    it('throws upgrade hint on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(fetchCorpusEpisodes('/c', {})).rejects.toThrow(
        /Corpus Library endpoint not found/,
      )
    })

    it('throws response text on a non-404 error', async () => {
      mockFetchJson(false, {}, 'boom', 500)
      await expect(fetchCorpusEpisodes('/c', {})).rejects.toThrow('boom')
    })
  })

  describe('fetchCorpusEpisodeDetail', () => {
    it('GETs detail with path and encoded metadata_relpath', async () => {
      const payload = {
        path: '/r',
        metadata_relative_path: 'metadata/a.metadata.json',
        feed_id: 'f',
        episode_id: 'e',
        episode_title: 'T',
        publish_date: null,
        summary_title: null,
        summary_bullets: [],
        summary_text: null,
        gi_relative_path: 'metadata/a.gi.json',
        kg_relative_path: 'metadata/a.kg.json',
        has_gi: true,
        has_kg: false,
      }
      mockFetchJson(true, payload)
      await expect(
        fetchCorpusEpisodeDetail('/root', 'metadata/a.metadata.json'),
      ).resolves.toEqual(payload)
      expectFetchCalledWithUrl(
        '/api/corpus/episodes/detail?path=%2Froot&metadata_relpath=metadata%2Fa.metadata.json',
      )
    })

    it('throws upgrade hint on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(
        fetchCorpusEpisodeDetail('/root', 'metadata/a.metadata.json'),
      ).rejects.toThrow(/Corpus Library endpoint not found/)
    })

    it('throws response text on a non-404 error', async () => {
      mockFetchJson(false, {}, 'detail failed', 500)
      await expect(
        fetchCorpusEpisodeDetail('/root', 'metadata/a.metadata.json'),
      ).rejects.toThrow('detail failed')
    })
  })

  describe('fetchCorpusSimilarEpisodes', () => {
    it('GETs similar with path, metadata_relpath, optional top_k', async () => {
      const payload: CorpusSimilarEpisodesResponse = {
        path: '/r',
        source_metadata_relative_path: 'metadata/a.metadata.json',
        query_used: 'q',
        items: [],
        error: 'no_index',
        detail: null,
      }
      mockFetchJson(true, payload)
      await fetchCorpusSimilarEpisodes('/root', 'metadata/a.metadata.json', { topK: 5 })
      expectFetchCalledWithUrl(
        '/api/corpus/episodes/similar?path=%2Froot&metadata_relpath=metadata%2Fa.metadata.json&top_k=5',
      )
    })

    it('omits top_k when options.topK is not provided', async () => {
      const payload: CorpusSimilarEpisodesResponse = {
        path: '/r',
        source_metadata_relative_path: 'metadata/a.metadata.json',
        query_used: 'q',
        items: [],
        error: null,
        detail: null,
      }
      mockFetchJson(true, payload)
      await fetchCorpusSimilarEpisodes('/root', 'metadata/a.metadata.json')
      const called = vi.mocked(fetch).mock.calls[0][0] as string
      expect(called).not.toMatch(/top_k=/)
    })

    it('throws upgrade hint on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(
        fetchCorpusSimilarEpisodes('/root', 'metadata/a.metadata.json'),
      ).rejects.toThrow(/Corpus Library endpoint not found/)
    })

    it('throws response text on a non-404 error', async () => {
      mockFetchJson(false, {}, 'similar failed', 500)
      await expect(
        fetchCorpusSimilarEpisodes('/root', 'metadata/a.metadata.json'),
      ).rejects.toThrow('similar failed')
    })
  })

  describe('fetchResolveEpisodeArtifacts', () => {
    it('POSTs path and trimmed/filtered episode_ids and returns parsed body', async () => {
      const payload = {
        path: '/c',
        resolved: [
          {
            episode_id: 'e1',
            publish_date: '2024-01-01',
            gi_relative_path: 'metadata/e1.gi.json',
            kg_relative_path: 'metadata/e1.kg.json',
            bridge_relative_path: null,
          },
        ],
        missing_episode_ids: ['e2'],
      }
      mockFetchJson(true, payload)
      await expect(
        fetchResolveEpisodeArtifacts('  /c  ', ['  e1  ', '', '  ', 'e2']),
      ).resolves.toEqual(payload)
      expect(fetch).toHaveBeenCalledWith(
        '/api/corpus/resolve-episode-artifacts',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: '/c', episode_ids: ['e1', 'e2'] }),
          signal: expect.any(AbortSignal),
        }),
      )
    })

    it('rejects when corpus path is blank without calling fetch', async () => {
      mockFetchJson(true, {})
      await expect(fetchResolveEpisodeArtifacts('   ', ['e1'])).rejects.toThrow(
        'Corpus path is required',
      )
      expect(fetch).not.toHaveBeenCalled()
    })

    it('rejects when no valid episode ids remain after filtering', async () => {
      mockFetchJson(true, {})
      await expect(fetchResolveEpisodeArtifacts('/c', ['  ', ''])).rejects.toThrow(
        'At least one episode id is required',
      )
      expect(fetch).not.toHaveBeenCalled()
    })

    it('throws upgrade hint on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(fetchResolveEpisodeArtifacts('/c', ['e1'])).rejects.toThrow(
        /Corpus Library endpoint not found/,
      )
    })

    it('throws response text on a non-404 error', async () => {
      mockFetchJson(false, {}, 'resolve failed', 500)
      await expect(fetchResolveEpisodeArtifacts('/c', ['e1'])).rejects.toThrow(
        'resolve failed',
      )
    })
  })

  describe('fetchNodeEpisodes', () => {
    it('POSTs /api/corpus/node-episodes with path and node_id', async () => {
      const payload = {
        path: '/c',
        node_id: 'topic:x',
        episodes: [],
        truncated: false,
        total_matched: null,
      }
      mockFetchJson(true, payload)
      await expect(fetchNodeEpisodes('/c', 'g:topic:x')).resolves.toEqual(payload)
      expect(fetch).toHaveBeenCalledWith(
        '/api/corpus/node-episodes',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: '/c', node_id: 'g:topic:x' }),
          signal: expect.any(AbortSignal),
        }),
      )
    })

    it('includes max_episodes when set', async () => {
      const payload = {
        path: '/c',
        node_id: 'topic:x',
        episodes: [],
        truncated: true,
        total_matched: 5,
      }
      mockFetchJson(true, payload)
      await fetchNodeEpisodes('/c', 'topic:x', 2)
      expect(JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string)).toEqual(
        { path: '/c', node_id: 'topic:x', max_episodes: 2 },
      )
    })

    it('includes default graph expand max when passing GRAPH_NODE_EPISODES_EXPAND_MAX', async () => {
      const payload = {
        path: '/c',
        node_id: 'topic:x',
        episodes: [],
        truncated: false,
        total_matched: null,
      }
      mockFetchJson(true, payload)
      await fetchNodeEpisodes('/c', 'topic:x', GRAPH_NODE_EPISODES_EXPAND_MAX)
      expect(JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string)).toEqual({
        path: '/c',
        node_id: 'topic:x',
        max_episodes: GRAPH_NODE_EPISODES_EXPAND_MAX,
      })
    })

    it('omits max_episodes when maxEpisodes is zero or negative', async () => {
      const payload = {
        path: '/c',
        node_id: 'topic:x',
        episodes: [],
        truncated: false,
        total_matched: null,
      }
      mockFetchJson(true, payload)
      await fetchNodeEpisodes('/c', 'topic:x', 0)
      expect(
        JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string),
      ).toEqual({ path: '/c', node_id: 'topic:x' })

      mockFetchJson(true, payload)
      await fetchNodeEpisodes('/c', 'topic:x', -5)
      expect(
        JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string),
      ).toEqual({ path: '/c', node_id: 'topic:x' })
    })

    it('trims node_id before sending', async () => {
      const payload = {
        path: '/c',
        node_id: 'topic:x',
        episodes: [],
        truncated: false,
        total_matched: null,
      }
      mockFetchJson(true, payload)
      await fetchNodeEpisodes('/c', '  topic:x  ')
      expect(
        JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string),
      ).toEqual({ path: '/c', node_id: 'topic:x' })
    })

    it('rejects when corpus path is blank without calling fetch', async () => {
      mockFetchJson(true, {})
      await expect(fetchNodeEpisodes('   ', 'topic:x')).rejects.toThrow(
        'Corpus path is required',
      )
      expect(fetch).not.toHaveBeenCalled()
    })

    it('rejects when node_id is blank without calling fetch', async () => {
      mockFetchJson(true, {})
      await expect(fetchNodeEpisodes('/c', '   ')).rejects.toThrow('node_id is required')
      expect(fetch).not.toHaveBeenCalled()
    })

    it('throws upgrade hint on 404', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(fetchNodeEpisodes('/c', 'topic:x')).rejects.toThrow(
        /Corpus Library endpoint not found/,
      )
    })

    it('throws response text on a non-404 error', async () => {
      mockFetchJson(false, {}, 'node failed', 500)
      await expect(fetchNodeEpisodes('/c', 'topic:x')).rejects.toThrow('node failed')
    })
  })
})
