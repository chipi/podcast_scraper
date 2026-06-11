import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  fetchCorpusManifest,
  fetchCorpusRunSummaryDocument,
  fetchCorpusRunsSummary,
  fetchCorpusStats,
} from './corpusMetricsApi'

function expectFetchCalledWithUrl(expectedUrl: string): void {
  expect(fetch).toHaveBeenCalledWith(
    expectedUrl,
    expect.objectContaining({
      signal: expect.any(AbortSignal),
    }),
  )
}

/** Stub ``fetch`` to return a single response. ``status`` defaults from ``ok``. */
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

describe('corpusMetricsApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  describe('fetchCorpusStats', () => {
    it('GETs /api/corpus/stats with trimmed path query and parses body', async () => {
      const payload = {
        path: '/resolved',
        publish_month_histogram: { '2024-01': 3 },
        catalog_episode_count: 12,
        catalog_feed_count: 4,
        digest_topics_configured: 2,
      }
      mockFetchJson(true, payload)
      await expect(fetchCorpusStats('  /my/corpus  ')).resolves.toEqual(payload)
      expectFetchCalledWithUrl('/api/corpus/stats?path=%2Fmy%2Fcorpus')
    })

    it('throws with response text when not ok', async () => {
      mockFetchJson(false, {}, 'missing corpus')
      await expect(fetchCorpusStats('/x')).rejects.toThrow('missing corpus')
    })

    it('throws HTTP status when not ok and empty body', async () => {
      mockFetchJson(false, {}, '', 503)
      await expect(fetchCorpusStats('/x')).rejects.toThrow('HTTP 503')
    })

    it('dedupes concurrent fetchCorpusStats for the same path into one HTTP call', async () => {
      mockFetchJson(true, {
        path: '/c',
        publish_month_histogram: {},
        catalog_episode_count: 1,
        catalog_feed_count: 1,
        digest_topics_configured: 0,
      })
      const [a, b] = await Promise.all([fetchCorpusStats('/c'), fetchCorpusStats('/c')])
      expect(a.catalog_episode_count).toBe(1)
      expect(b.catalog_episode_count).toBe(1)
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
      expectFetchCalledWithUrl('/api/corpus/stats?path=%2Fc')
    })

    it('does not dedupe sequential calls (no response cache)', async () => {
      mockFetchJson(true, {
        path: '/c',
        publish_month_histogram: {},
        catalog_episode_count: 1,
        catalog_feed_count: 1,
        digest_topics_configured: 0,
      })
      await fetchCorpusStats('/seq')
      await fetchCorpusStats('/seq')
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2)
    })

    it('propagates network errors from fetch', async () => {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => {
          throw new Error('network down')
        }) as unknown as typeof fetch,
      )
      await expect(fetchCorpusStats('/x')).rejects.toThrow('network down')
    })
  })

  describe('fetchCorpusRunsSummary', () => {
    it('GETs /api/corpus/runs/summary with trimmed path and parses runs', async () => {
      const payload = {
        path: '/resolved',
        runs: [
          {
            relative_path: 'r/1',
            run_id: 'abc',
            created_at: null,
            run_duration_seconds: null,
            episodes_scraped_total: 5,
            errors_total: 0,
            gi_artifacts_generated: 5,
            kg_artifacts_generated: 5,
            time_scraping_seconds: null,
            time_parsing_seconds: null,
            time_normalizing_seconds: null,
            time_io_and_waiting_seconds: null,
            episode_outcomes: { ok: 5 },
            ads_filtered_count: null,
            dialogue_insights_dropped_count: null,
            topics_normalized_count: null,
            entity_kinds_repaired_count: null,
            ad_chars_excised_preroll: null,
            ad_chars_excised_postroll: null,
            ad_episodes_with_excision_count: null,
          },
        ],
      }
      mockFetchJson(true, payload)
      await expect(fetchCorpusRunsSummary('  /c  ')).resolves.toEqual(payload)
      expectFetchCalledWithUrl('/api/corpus/runs/summary?path=%2Fc')
    })

    it('throws with response text when not ok', async () => {
      mockFetchJson(false, {}, 'no runs')
      await expect(fetchCorpusRunsSummary('/x')).rejects.toThrow('no runs')
    })

    it('dedupes concurrent calls for the same path', async () => {
      mockFetchJson(true, { path: '/c', runs: [] })
      const [a, b] = await Promise.all([
        fetchCorpusRunsSummary('/c'),
        fetchCorpusRunsSummary('/c'),
      ])
      expect(a.runs).toEqual([])
      expect(b.runs).toEqual([])
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    })
  })

  describe('fetchCorpusManifest', () => {
    it('GETs /api/corpus/documents/manifest with trimmed path and parses document', async () => {
      const payload = {
        schema_version: 'v1',
        feeds: [
          {
            feed_url: 'http://f',
            stable_feed_dir: 'd',
            episodes_processed: 3,
            last_run_finished_at: '2024-01-01',
            ok: true,
          },
        ],
      }
      mockFetchJson(true, payload)
      await expect(fetchCorpusManifest('  /c  ')).resolves.toEqual(payload)
      expectFetchCalledWithUrl('/api/corpus/documents/manifest?path=%2Fc')
    })

    it('throws HTTP status when not ok and empty body', async () => {
      mockFetchJson(false, {}, '', 404)
      await expect(fetchCorpusManifest('/x')).rejects.toThrow('HTTP 404')
    })

    it('dedupes concurrent calls for the same path', async () => {
      mockFetchJson(true, { schema_version: 'v1', feeds: [] })
      const [a, b] = await Promise.all([
        fetchCorpusManifest('/c'),
        fetchCorpusManifest('/c'),
      ])
      expect(a).toEqual({ schema_version: 'v1', feeds: [] })
      expect(b).toEqual({ schema_version: 'v1', feeds: [] })
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    })
  })

  describe('fetchCorpusRunSummaryDocument', () => {
    it('GETs /api/corpus/documents/run-summary with trimmed path and returns raw doc', async () => {
      const payload = { arbitrary: 'shape', nested: { count: 7 } }
      mockFetchJson(true, payload)
      await expect(fetchCorpusRunSummaryDocument('  /c  ')).resolves.toEqual(payload)
      expectFetchCalledWithUrl('/api/corpus/documents/run-summary?path=%2Fc')
    })

    it('throws with response text when not ok', async () => {
      mockFetchJson(false, {}, 'no summary')
      await expect(fetchCorpusRunSummaryDocument('/x')).rejects.toThrow('no summary')
    })

    it('dedupes concurrent calls for the same path', async () => {
      mockFetchJson(true, { ok: true })
      const [a, b] = await Promise.all([
        fetchCorpusRunSummaryDocument('/c'),
        fetchCorpusRunSummaryDocument('/c'),
      ])
      expect(a).toEqual({ ok: true })
      expect(b).toEqual({ ok: true })
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    })
  })
})
