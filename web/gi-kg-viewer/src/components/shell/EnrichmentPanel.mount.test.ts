// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

import EnrichmentPanel from './EnrichmentPanel.vue'

/**
 * RFC-088 chunk-9 follow-up — real mount tests for the Configuration
 * popup's Enrichment tab. Goes beyond static-source guards: actually
 * renders the component against stubbed fetch responses and asserts
 * the user-visible behaviour (counts, row rendering, button wiring,
 * notice text after submit).
 */

interface FetchCall {
  url: string
  init?: RequestInit
}

const calls: FetchCall[] = []

function jsonRes(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFetch(routes: Record<string, unknown | ((init?: RequestInit) => unknown)>): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      calls.push({ url, init })
      for (const [key, value] of Object.entries(routes)) {
        if (url.includes(key)) {
          const body = typeof value === 'function' ? (value as (i?: RequestInit) => unknown)(init) : value
          return jsonRes(body)
        }
      }
      return jsonRes({ ok: true })
    }),
  )
}

beforeEach(() => {
  calls.length = 0
})

afterEach(() => {
  vi.unstubAllGlobals()
})


describe('EnrichmentPanel — mount + behaviour', () => {
  it('renders the empty state when no enrichers seen yet', async () => {
    stubFetch({
      '/api/enrichment/status': { available: false, reason: 'no status yet' },
      '/api/enrichment/health': { enrichers: {} },
      '/api/enrichment/metrics': { window: '24h', per_enricher: {} },
      '/api/enrichment/run-summary': { available: false },
      '/api/corpus/enrichments?': { enrichments: [] },
    })
    const w = mount(EnrichmentPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    expect(w.get('[data-testid="enrichment-row-count"]').text()).toBe('0')
    expect(w.get('[data-testid="enrichment-total-runs"]').text()).toBe('0')
    expect(w.get('[data-testid="enrichment-autodisabled-count"]').text()).toBe('0')
    expect(w.find('[data-testid="enrichment-empty-row"]').exists()).toBe(true)
  })

  it('renders one row per known enricher and totals correctly', async () => {
    stubFetch({
      '/api/enrichment/status': { available: false },
      '/api/enrichment/health': {
        enrichers: {
          topic_similarity: {
            consecutive_failures: 0,
            auto_disabled: false,
            circuit_state: 'closed',
            last_status: 'ok',
          },
          nli_contradiction: {
            consecutive_failures: 3,
            auto_disabled: true,
            circuit_state: 'open',
            last_status: 'failed',
            auto_disabled_reason: 'transient HF outage',
          },
        },
      },
      '/api/enrichment/metrics': {
        window: '24h',
        per_enricher: {
          topic_similarity: { runs_total: 5, runs_ok: 5, runs_failed: 0 },
          nli_contradiction: { runs_total: 4, runs_ok: 1, runs_failed: 3 },
        },
      },
      '/api/enrichment/run-summary': { available: false },
      '/api/corpus/enrichments?': { enrichments: [] },
    })
    const w = mount(EnrichmentPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    expect(w.get('[data-testid="enrichment-row-count"]').text()).toBe('2')
    expect(w.get('[data-testid="enrichment-total-runs"]').text()).toBe('9')  // 5 + 4
    expect(w.get('[data-testid="enrichment-autodisabled-count"]').text()).toBe('1')
    // Each row rendered.
    expect(w.find('[data-testid="enrichment-row-topic_similarity"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-row-nli_contradiction"]').exists()).toBe(true)
    // Re-enable button appears ONLY on auto_disabled rows.
    expect(w.find('[data-testid="enrichment-re-enable-topic_similarity"]').exists()).toBe(false)
    expect(w.find('[data-testid="enrichment-re-enable-nli_contradiction"]').exists()).toBe(true)
  })

  it('Run enrichment now → POSTs /api/jobs/enrichment and shows the notice', async () => {
    stubFetch({
      '/api/enrichment/status': { available: false },
      '/api/enrichment/health': { enrichers: {} },
      '/api/enrichment/metrics': { window: '24h', per_enricher: {} },
      '/api/enrichment/run-summary': { available: false },
      '/api/corpus/enrichments?': { enrichments: [] },
      '/api/jobs/enrichment': () => ({
        job_id: 'job-deadbeef-1234',
        status: 'running',
        corpus_path: '/c',
      }),
    })
    const w = mount(EnrichmentPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    await w.get('[data-testid="enrichment-run-btn"]').trigger('click')
    await flushPromises()
    // Submit hit the right URL with POST.
    const submit = calls.find((c) => c.url.includes('/api/jobs/enrichment'))
    expect(submit).toBeTruthy()
    expect(submit!.init?.method).toBe('POST')
    // User-visible notice.
    const notice = w.get('[data-testid="enrichment-submit-notice"]').text()
    expect(notice).toContain('job-deadbeef')
    expect(notice).toContain('running')
  })

  it('Re-enable click → POSTs /api/enrichment/health/<id>/re-enable + refreshes', async () => {
    stubFetch({
      '/api/enrichment/status': { available: false },
      '/api/enrichment/health': {
        enrichers: {
          nli_contradiction: {
            consecutive_failures: 3,
            auto_disabled: true,
            circuit_state: 'open',
            last_status: 'failed',
          },
        },
      },
      '/api/enrichment/metrics': { window: '24h', per_enricher: {} },
      '/api/enrichment/run-summary': { available: false },
      '/api/corpus/enrichments?': { enrichments: [] },
      '/api/enrichment/health/nli_contradiction/re-enable': () => ({
        enricher_id: 'nli_contradiction',
        auto_disabled: false,
        consecutive_failures: 0,
      }),
    })
    const w = mount(EnrichmentPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    await w.get('[data-testid="enrichment-re-enable-nli_contradiction"]').trigger('click')
    await flushPromises()
    const reEnable = calls.find((c) =>
      c.url.includes('/api/enrichment/health/nli_contradiction/re-enable'),
    )
    expect(reEnable).toBeTruthy()
    expect(reEnable!.init?.method).toBe('POST')
    // After re-enable, the panel refreshes — fetch must have hit /api/enrichment/health twice
    // (initial load + post-re-enable refresh).
    const healthHits = calls.filter((c) => c.url.includes('/api/enrichment/health?')).length
    expect(healthHits).toBe(2)
  })

  it('Refresh click re-runs the load with no submit', async () => {
    stubFetch({
      '/api/enrichment/status': { available: false },
      '/api/enrichment/health': { enrichers: {} },
      '/api/enrichment/metrics': { window: '24h', per_enricher: {} },
      '/api/enrichment/run-summary': { available: false },
      '/api/corpus/enrichments?': { enrichments: [] },
    })
    const w = mount(EnrichmentPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    const before = calls.filter((c) => c.url.includes('/api/enrichment/status')).length
    await w.get('[data-testid="enrichment-refresh-btn"]').trigger('click')
    await flushPromises()
    const after = calls.filter((c) => c.url.includes('/api/enrichment/status')).length
    expect(after).toBeGreaterThan(before)
    // No POST to /api/jobs/enrichment during refresh.
    expect(calls.find((c) => c.url.includes('/api/jobs/enrichment'))).toBeUndefined()
  })

  it('renders error banner when an API call fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonRes({ detail: 'boom' }, 500)),
    )
    const w = mount(EnrichmentPanel, { props: { corpusPath: '/c' } })
    await flushPromises()
    expect(w.text()).toContain('boom')
  })
})
