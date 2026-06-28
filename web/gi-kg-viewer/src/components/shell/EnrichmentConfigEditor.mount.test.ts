// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import EnrichmentConfigEditor from './EnrichmentConfigEditor.vue'

interface FetchCall {
  url: string
  init?: RequestInit
}
const fetchCalls: FetchCall[] = []

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const schemaResponse = {
  type: 'object',
  properties: {
    enrichers: {
      properties: {
        temporal_velocity: {
          properties: {
            alpha: { type: 'number', minimum: 0, maximum: 1, default: 0.5 },
            window_months: { type: 'integer', minimum: 1, maximum: 36, default: 12 },
          },
        },
        topic_similarity: {
          properties: {
            top_k: { type: 'integer', minimum: 1, maximum: 100, default: 10 },
            provider: {
              oneOf: [
                {
                  type: 'object',
                  properties: {
                    type: { const: 'fake_for_test', description: 'fake' },
                    dim: { type: 'integer', default: 32 },
                  },
                },
              ],
            },
          },
        },
      },
    },
  },
}

const providerTypesResponse = {
  by_protocol: {
    EmbeddingProvider: [
      {
        name: 'fake_for_test',
        protocol: 'EmbeddingProvider',
        description: 'Deterministic fake.',
        params_schema: {
          properties: { dim: { type: 'integer', default: 32 } },
        },
      },
    ],
    NliScorer: [],
  },
}

let configResponse: Record<string, unknown> = {
  corpus_path: '/corpus',
  profile: null,
  profile_block: {},
  operator_block: {},
  resolved_block: {},
}

beforeEach(() => {
  fetchCalls.length = 0
  configResponse = {
    corpus_path: '/corpus',
    profile: null,
    profile_block: {},
    operator_block: {},
    resolved_block: {},
  }
  vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
    fetchCalls.push({ url, init })
    if (url.includes('/api/enrichment/config/schema'))
      return Promise.resolve(jsonResponse(schemaResponse))
    if (url.includes('/api/enrichment/provider-types'))
      return Promise.resolve(jsonResponse(providerTypesResponse))
    if (url.includes('/api/enrichment/config')) {
      if (init?.method === 'PUT') {
        const body = JSON.parse(String(init.body))
        configResponse = {
          ...configResponse,
          operator_block: body.enrichment_block,
          resolved_block: body.enrichment_block,
        }
      }
      return Promise.resolve(jsonResponse(configResponse))
    }
    return Promise.resolve(new Response('', { status: 404 }))
  })
})

async function mountEditor() {
  const w = mount(EnrichmentConfigEditor, {
    attachTo: document.body,
    props: { corpusPath: '/corpus' },
  })
  for (let i = 0; i < 6; i++) await w.vm.$nextTick()
  return w
}

describe('EnrichmentConfigEditor.vue', () => {
  it('fetches config + schema + provider-types on mount and renders rows', async () => {
    const w = await mountEditor()
    const urls = fetchCalls.map((c) => c.url)
    expect(urls.some((u) => u.includes('/api/enrichment/config?'))).toBe(true)
    expect(urls.some((u) => u.endsWith('/api/enrichment/config/schema'))).toBe(true)
    expect(urls.some((u) => u.endsWith('/api/enrichment/provider-types'))).toBe(true)
    expect(w.find('[data-testid="enrichment-config-row-temporal_velocity"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-config-row-topic_similarity"]').exists()).toBe(true)
  })

  it('renders the global on/off + per-row enabled checkboxes', async () => {
    const w = await mountEditor()
    expect(w.find('[data-testid="enrichment-config-global-enabled"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-config-enabled-temporal_velocity"]').exists()).toBe(true)
    expect(w.find('[data-testid="enrichment-config-enabled-topic_similarity"]').exists()).toBe(true)
  })

  it('renders knob inputs derived from manifest config_schema', async () => {
    const w = await mountEditor()
    expect(
      w.find('[data-testid="enrichment-config-knob-temporal_velocity-alpha"]').exists(),
    ).toBe(true)
    expect(
      w.find('[data-testid="enrichment-config-knob-temporal_velocity-window_months"]').exists(),
    ).toBe(true)
  })

  it('renders provider dropdown for ML enrichers with the right options', async () => {
    const w = await mountEditor()
    const sel = w.find<HTMLSelectElement>(
      '[data-testid="enrichment-config-provider-type-topic_similarity"]',
    )
    expect(sel.exists()).toBe(true)
    const optionValues = sel
      .findAll('option')
      .map((o) => (o.element as HTMLOptionElement).value)
    expect(optionValues).toContain('')
    expect(optionValues).toContain('fake_for_test')
  })

  it('Save sends PUT with the operator block; reload reflects persisted state', async () => {
    const w = await mountEditor()
    // Edit alpha
    const alpha = w.find<HTMLInputElement>(
      '[data-testid="enrichment-config-knob-temporal_velocity-alpha"]',
    )
    await alpha.setValue('0.7')
    // Save
    const saveBtn = w.find('[data-testid="enrichment-config-save-btn"]')
    expect(saveBtn.attributes('disabled')).toBeUndefined()
    await saveBtn.trigger('click')
    for (let i = 0; i < 6; i++) await w.vm.$nextTick()
    const puts = fetchCalls.filter((c) => c.init?.method === 'PUT')
    expect(puts.length).toBe(1)
    const body = JSON.parse(String(puts[0].init?.body))
    expect(body.enrichment_block.enrichers.temporal_velocity.alpha).toBe(0.7)
  })
})
