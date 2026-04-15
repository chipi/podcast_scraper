import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchCorpusStats } from './corpusMetricsApi'

function expectFetchCalledWithUrl(expectedUrl: string): void {
  expect(fetch).toHaveBeenCalledWith(
    expectedUrl,
    expect.objectContaining({
      signal: expect.any(AbortSignal),
    }),
  )
}

describe('corpusMetricsApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('dedupes concurrent fetchCorpusStats for the same path into one HTTP call', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        return {
          ok: true,
          json: async () => ({
            path: '/c',
            publish_month_histogram: {},
            catalog_episode_count: 1,
            catalog_feed_count: 1,
            digest_topics_configured: 0,
          }),
        }
      }) as unknown as typeof fetch,
    )

    const p1 = fetchCorpusStats('/c')
    const p2 = fetchCorpusStats('/c')
    const [a, b] = await Promise.all([p1, p2])

    expect(a.catalog_episode_count).toBe(1)
    expect(b.catalog_episode_count).toBe(1)
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
    expectFetchCalledWithUrl('/api/corpus/stats?path=%2Fc')
  })
})
