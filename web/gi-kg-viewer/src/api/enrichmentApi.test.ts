import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getCorpusEnrichmentEnvelope,
  getCorpusEnrichmentsCatalogue,
  getEnrichmentEvents,
  getEnrichmentHealth,
  getEnrichmentMetrics,
  getEnrichmentRunSummary,
  getEnrichmentStatus,
  reEnableEnricher,
  submitEnrichmentJob,
} from './enrichmentApi'

/**
 * RFC-088 chunk-8 follow-up: cover the TypeScript-side enrichment API
 * client. Stubs global fetch (which fetchWithTimeout calls into) so
 * every helper's URL + params + body are asserted against the
 * server-side route shapes documented in
 * ``docs/api/ENRICHMENT_LAYER_API.md``.
 */

interface FetchCall {
  url: string
  init?: RequestInit
}

const calls: FetchCall[] = []

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

beforeEach(() => {
  calls.length = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push({ url: String(input), init })
      return jsonResponse({ ok: true })
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('enrichmentApi — URL + param shape', () => {
  it('getEnrichmentStatus hits /api/enrichment/status', async () => {
    await getEnrichmentStatus('/path/to/corpus')
    expect(calls[0]?.url).toContain('/api/enrichment/status')
    expect(calls[0]?.url).toContain('path=%2Fpath%2Fto%2Fcorpus')
  })

  it('getEnrichmentHealth hits /api/enrichment/health', async () => {
    await getEnrichmentHealth('/c')
    expect(calls[0]?.url).toContain('/api/enrichment/health')
  })

  it('getEnrichmentMetrics defaults window=24h', async () => {
    await getEnrichmentMetrics('/c')
    expect(calls[0]?.url).toContain('window=24h')
  })

  it('getEnrichmentMetrics respects custom window', async () => {
    await getEnrichmentMetrics('/c', '1h')
    expect(calls[0]?.url).toContain('window=1h')
  })

  it('getEnrichmentRunSummary hits /api/enrichment/run-summary', async () => {
    await getEnrichmentRunSummary('/c')
    expect(calls[0]?.url).toContain('/api/enrichment/run-summary')
  })

  it('getEnrichmentEvents threads enricher_id / event_type / limit', async () => {
    await getEnrichmentEvents('/c', {
      enricher_id: 'topic_similarity',
      event_type: 'enrichment.enricher.completed',
      limit: 20,
    })
    const url = calls[0]?.url ?? ''
    expect(url).toContain('enricher_id=topic_similarity')
    expect(url).toContain('event_type=enrichment.enricher.completed')
    expect(url).toContain('limit=20')
  })

  it('getCorpusEnrichmentsCatalogue hits /api/corpus/enrichments', async () => {
    await getCorpusEnrichmentsCatalogue('/c')
    expect(calls[0]?.url).toContain('/api/corpus/enrichments?')
  })

  it('getCorpusEnrichmentEnvelope hits /api/corpus/enrichments/<id>', async () => {
    await getCorpusEnrichmentEnvelope('/c', 'topic_similarity')
    expect(calls[0]?.url).toContain('/api/corpus/enrichments/topic_similarity')
  })

  it('getCorpusEnrichmentEnvelope returns null on 404', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(null, { status: 404 })),
    )
    const r = await getCorpusEnrichmentEnvelope('/c', 'never_ran')
    expect(r).toBeNull()
  })

  it('reEnableEnricher POSTs to /api/enrichment/health/<id>/re-enable with body', async () => {
    await reEnableEnricher('/c', 'topic_similarity', 'transient outage')
    const call = calls[0]
    expect(call?.url).toContain('/api/enrichment/health/topic_similarity/re-enable')
    expect(call?.init?.method).toBe('POST')
    expect(call?.init?.body).toBe(JSON.stringify({ reason: 'transient outage' }))
  })

  it('reEnableEnricher uses a default reason when omitted', async () => {
    await reEnableEnricher('/c', 'x')
    expect(calls[0]?.init?.body).toContain('operator re-enable from viewer')
  })

  it('submitEnrichmentJob POSTs to /api/jobs/enrichment with body shape', async () => {
    await submitEnrichmentJob('/c', {
      only: ['topic_similarity'],
      skip: [],
      corpus_only: true,
    })
    const call = calls[0]
    expect(call?.url).toContain('/api/jobs/enrichment')
    expect(call?.init?.method).toBe('POST')
    const parsed = JSON.parse(String(call?.init?.body))
    expect(parsed.only).toEqual(['topic_similarity'])
    expect(parsed.corpus_only).toBe(true)
  })

  it('submitEnrichmentJob defaults to empty body', async () => {
    await submitEnrichmentJob('/c')
    expect(calls[0]?.init?.body).toBe(JSON.stringify({}))
  })
})

describe('enrichmentApi — error path', () => {
  it('throws on non-2xx (other than 404 for envelope GETs)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse({ detail: 'boom' }, 500)),
    )
    await expect(getEnrichmentStatus('/c')).rejects.toThrow()
  })
})
