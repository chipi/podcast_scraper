// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import EpisodeEnrichmentSection from './EpisodeEnrichmentSection.vue'

/**
 * RFC-088 chunk-9 follow-up — mount tests for the episode detail
 * panel's enrichment section. Verifies it reads the insight_density
 * envelope, renders the density bars, and hides itself when absent.
 * (Per-episode topic co-occurrence was removed — it lives at corpus
 * scope on the Topic node card.)
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFetch(payloads: Record<string, unknown>): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      for (const [key, body] of Object.entries(payloads)) {
        if (url.includes(key)) return res(body)
      }
      return res({}, 404)
    }),
  )
}

beforeEach(() => {
  setActivePinia(createPinia())
})

afterEach(() => {
  vi.unstubAllGlobals()
})


describe('EpisodeEnrichmentSection — mount + behaviour', () => {
  it('hides itself when both envelopes return 404', async () => {
    stubFetch({})
    const w = mount(EpisodeEnrichmentSection, {
      props: { corpusPath: '/c', metadataRelpath: 'metadata/ep1.metadata.json' },
    })
    await flushPromises()
    expect(w.find('[data-testid="episode-enrichment-section"]').exists()).toBe(false)
  })

  it('renders the density bars when insight_density present', async () => {
    stubFetch({
      'episode/enrichments/insight_density': {
        data: {
          episode_id: 'episode:ep1',
          duration_seconds: 900,
          has_timing: true,
          counts: { early: 3, mid: 2, late: 1 },
          total_insights: 6,
        },
      },
    })
    const w = mount(EpisodeEnrichmentSection, {
      props: { corpusPath: '/c', metadataRelpath: 'metadata/ep1.metadata.json' },
    })
    await flushPromises()
    expect(w.find('[data-testid="episode-enrichment-section"]').exists()).toBe(true)
    expect(w.find('[data-testid="episode-enrichment-density"]').exists()).toBe(true)
    const text = w.get('[data-testid="episode-enrichment-density"]').text()
    expect(text).toContain('6 insights')
    expect(text).toContain('early')
    expect(text).toContain('mid')
    expect(text).toContain('late')
  })

  it('shows the "no timing — even split" hint when has_timing is false', async () => {
    stubFetch({
      'episode/enrichments/insight_density': {
        data: {
          has_timing: false,
          counts: { early: 1, mid: 1, late: 1 },
          total_insights: 3,
        },
      },
    })
    const w = mount(EpisodeEnrichmentSection, {
      props: { corpusPath: '/c', metadataRelpath: 'metadata/ep1.metadata.json' },
    })
    await flushPromises()
    expect(w.text()).toContain('no timing')
  })
})
