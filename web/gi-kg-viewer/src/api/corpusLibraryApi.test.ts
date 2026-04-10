import { afterEach, describe, expect, it, vi } from 'vitest'
import type { CorpusSimilarEpisodesResponse } from './corpusLibraryApi'
import {
  fetchCorpusEpisodeDetail,
  fetchCorpusEpisodes,
  fetchCorpusFeeds,
  fetchCorpusSimilarEpisodes,
} from './corpusLibraryApi'

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
      expect(fetch).toHaveBeenCalledWith('/api/corpus/feeds?path=%2Fmy%2Fcorpus')
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
  })

  describe('fetchCorpusEpisodes', () => {
    it('omits feed_id when options.feedId is undefined', async () => {
      const payload = { path: '/r', feed_id: null, items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', {})
      expect(fetch).toHaveBeenCalledWith('/api/corpus/episodes?path=%2Fc')
    })

    it('includes feed_id for empty string (ungrouped filter)', async () => {
      const payload = { path: '/r', feed_id: '', items: [], next_cursor: null }
      mockFetchJson(true, payload)
      await fetchCorpusEpisodes('/c', { feedId: '' })
      expect(fetch).toHaveBeenCalledWith(
        '/api/corpus/episodes?path=%2Fc&feed_id=',
      )
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
      expect(fetch).toHaveBeenCalledWith(
        '/api/corpus/episodes/detail?path=%2Froot&metadata_relpath=metadata%2Fa.metadata.json',
      )
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
      expect(fetch).toHaveBeenCalledWith(
        '/api/corpus/episodes/similar?path=%2Froot&metadata_relpath=metadata%2Fa.metadata.json&top_k=5',
      )
    })
  })
})
