import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  fetchCachedCorpusEnvelope,
  invalidateEnrichmentCache,
} from './useEnrichmentEnvelopeCache'

/**
 * RFC-088 chunk-8 follow-up: the rail-side enrichment cache. We stub the
 * underlying fetch helper and assert that:
 *   - the same (corpus, enricherId) is fetched once, then served from cache
 *   - invalidate({corpusPath}) drops all entries for that corpus
 *   - invalidate({corpusPath, enricherId}) drops only the matching entry
 *   - a rejected fetch is removed from the cache so the next call retries
 */

let calls = 0

beforeEach(() => {
  invalidateEnrichmentCache()
  calls = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      calls += 1
      return new Response(
        JSON.stringify({ schema_version: '1.0', enricher_id: 'x', data: {} }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useEnrichmentEnvelopeCache', () => {
  it('caches per (corpusPath, enricherId)', async () => {
    await fetchCachedCorpusEnvelope('/c', 'x')
    await fetchCachedCorpusEnvelope('/c', 'x')
    expect(calls).toBe(1)
  })

  it('refetches when enricherId changes', async () => {
    await fetchCachedCorpusEnvelope('/c', 'x')
    await fetchCachedCorpusEnvelope('/c', 'y')
    expect(calls).toBe(2)
  })

  it('refetches when corpusPath changes', async () => {
    await fetchCachedCorpusEnvelope('/c1', 'x')
    await fetchCachedCorpusEnvelope('/c2', 'x')
    expect(calls).toBe(2)
  })

  it('invalidate({corpusPath}) drops all entries for that corpus', async () => {
    await fetchCachedCorpusEnvelope('/c', 'a')
    await fetchCachedCorpusEnvelope('/c', 'b')
    invalidateEnrichmentCache({ corpusPath: '/c' })
    await fetchCachedCorpusEnvelope('/c', 'a')
    expect(calls).toBe(3)
  })

  it('invalidate({corpusPath, enricherId}) drops only the matching entry', async () => {
    await fetchCachedCorpusEnvelope('/c', 'a')
    await fetchCachedCorpusEnvelope('/c', 'b')
    invalidateEnrichmentCache({ corpusPath: '/c', enricherId: 'a' })
    await fetchCachedCorpusEnvelope('/c', 'a') // re-fetch
    await fetchCachedCorpusEnvelope('/c', 'b') // still cached
    expect(calls).toBe(3)
  })

  it('drops failing entries so the next call can retry', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        calls += 1
        throw new Error('network')
      }),
    )
    await expect(fetchCachedCorpusEnvelope('/c', 'flaky')).rejects.toThrow()
    await expect(fetchCachedCorpusEnvelope('/c', 'flaky')).rejects.toThrow()
    expect(calls).toBe(2) // retried, not served the broken promise
  })
})
