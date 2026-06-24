import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, Entity, Insight, Topic } from '../services/types'
import KnowledgePanel from './KnowledgePanel.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

const emptyPage = { items: [], page: 1, page_size: 6, total: 0, has_more: false }

beforeEach(() => {
  // Default: no related peers (index unavailable) so the section hides.
  vi.spyOn(api, 'getRelated').mockResolvedValue(emptyPage)
})
afterEach(() => vi.restoreAllMocks())

function episode(): EpisodeDetail {
  return {
    slug: 's1',
    title: 'Ep',
    feed_id: 'f',
    podcast_title: 'Show',
    publish_date: '2024-01-01',
    duration_seconds: 1800,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    summary_title: 'Sum',
    summary_bullets: [],
    summary_text: 'A short summary.',
    has_transcript: true,
    has_summary: true,
    has_gi: true,
    has_kg: true,
    has_bridge: false,
  }
}

function insight(over: Partial<Insight> = {}): Insight {
  return {
    id: 'i1',
    text: 'Sleep consolidates memory.',
    grounded: true,
    insight_type: 'claim',
    confidence: null,
    position_hint: null,
    quotes: [
      { text: 'the spindles gate memory', speaker: 'person:matthew-walker', char_start: null, char_end: null, start_ms: 12000, end_ms: 15000 },
    ],
    ...over,
  }
}

function mountPanel(props: Partial<Parameters<typeof KnowledgePanel>[0]> = {}) {
  return mount(KnowledgePanel, {
    props: {
      episode: episode(),
      insights: [insight()],
      topics: [{ id: 'topic:memory', label: 'memory' } as Topic],
      persons: [{ id: 'person:matthew-walker', name: 'Matthew Walker', kind: 'person' } as Entity],
      slug: 's1',
      activeInsightId: null,
      ...props,
    },
    global: { plugins: [i18n, router] },
  })
}

describe('KnowledgePanel', () => {
  it('renders summary, topics, people, and insight cards', () => {
    const w = mountPanel()
    expect(w.text()).toContain('A short summary.')
    expect(w.text()).toContain('memory')
    expect(w.text()).toContain('Matthew Walker')
    expect(w.text()).toContain('Sleep consolidates memory.')
    expect(w.text()).toContain('the spindles gate memory') // verbatim quote
  })

  it('emits seek with the insight quote start (jump-to-moment)', async () => {
    const w = mountPanel()
    // The timestamp button shows 0:12 (12000ms).
    const btn = w.findAll('button').find((b) => b.text().includes('0:12'))
    expect(btn).toBeTruthy()
    await btn!.trigger('click')
    expect(w.emitted('seek')?.[0]).toEqual([12])
  })

  it('filters insights by person and clears', async () => {
    const other = insight({ id: 'i2', text: 'Other claim.', quotes: [] })
    const w = mountPanel({ insights: [insight(), other] })
    expect(w.text()).toContain('Other claim.')
    // Tap the person → only insights with that speaker remain.
    await w.findAll('button').find((b) => b.text() === 'Matthew Walker')!.trigger('click')
    expect(w.text()).not.toContain('Other claim.')
    expect(w.text()).toContain('Sleep consolidates memory.')
  })

  it('runs episode-scoped search and renders grounded results', async () => {
    const api = await import('../services/api')
    vi.spyOn(api, 'searchEpisode').mockResolvedValue({
      query: 'memory',
      error: null,
      results: [
        {
          doc_id: 'd1',
          score: 0.9,
          text: 'A grounded passage about memory.',
          metadata: {},
          source_tier: 'segment',
          lifted: { quote: { timestamp_start_ms: 20000 } },
        },
      ],
    })
    const w = mountPanel()
    await w.find('input').setValue('memory')
    await w.find('form').trigger('submit')
    await new Promise((r) => setTimeout(r, 0))
    expect(w.text()).toContain('A grounded passage about memory.')
    const jump = w.findAll('button').find((b) => b.text().includes('0:20'))
    await jump!.trigger('click')
    expect(w.emitted('seek')?.at(-1)).toEqual([20])
  })

  it('renders "More like this" peers with links to the player', async () => {
    vi.spyOn(api, 'getRelated').mockResolvedValue({
      items: [
        {
          slug: 'peer-1', title: 'A Related Episode', feed_id: 'f', podcast_title: 'Show',
          publish_date: null, duration_seconds: null, episode_image_url: null, feed_image_url: null,
          artwork_url: null, status: 'ready', summary_preview: null, topics: [],
          has_transcript: true, has_summary: false, has_gi: false, has_kg: false, has_bridge: false,
        },
      ],
      page: 1, page_size: 6, total: 1, has_more: false,
    })
    const w = mountPanel()
    await flushPromises()
    expect(w.text()).toContain('More like this')
    expect(w.text()).toContain('A Related Episode')
    expect(w.findAll('a').map((a) => a.attributes('href'))).toContain('/episode/peer-1')
  })

  it('shows the empty message when no intelligence is present', () => {
    const e = episode()
    e.summary_text = null
    e.summary_title = null
    const w = mountPanel({ episode: e, insights: [], topics: [], persons: [] })
    expect(w.text()).toContain('Insights appear once this episode is processed.')
  })
})
