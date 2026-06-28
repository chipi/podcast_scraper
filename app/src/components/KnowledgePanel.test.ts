import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, Entity, Highlight, Insight, Topic } from '../services/types'
import { useAuthStore } from '../stores/auth'
import KnowledgePanel from './KnowledgePanel.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
  ],
})

const emptyPage = { items: [], page: 1, page_size: 6, total: 0, has_more: false }

beforeEach(() => {
  setActivePinia(createPinia()) // FavoriteButton (on insights) resolves the favorites/auth stores
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

  it('tapping a person chip opens its entity card (PRD-043)', async () => {
    const getPerson = vi.spyOn(api, 'getPersonCard').mockResolvedValue({
      id: 'person:matthew-walker',
      label: 'Matthew Walker',
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
    })
    const w = mountPanel()
    await w.findAll('button').find((b) => b.text() === 'Matthew Walker')!.trigger('click')
    await flushPromises()
    // Replace-in-panel (UXS-014): the card renders INLINE in the panel (no overlay), with a ‹ Back.
    expect(getPerson).toHaveBeenCalledWith('person:matthew-walker')
    expect(w.text()).toContain('Matthew Walker')
    expect(w.findAll('button').some((b) => b.text().includes('Back'))).toBe(true)
  })

  it('tapping a topic chip opens its entity card (not a search)', async () => {
    const getTopic = vi.spyOn(api, 'getTopicCard').mockResolvedValue({
      id: 'topic:memory',
      label: 'memory',
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
    const push = vi.spyOn(router, 'push')
    const w = mountPanel()
    await w.findAll('button').find((b) => b.text() === 'memory')!.trigger('click')
    await flushPromises()
    expect(getTopic).toHaveBeenCalledWith('topic:memory')
    expect(push).not.toHaveBeenCalled() // search now lives inside the card, not on chip-tap
  })

  it('orders topics cluster-first and marks the dominant cluster (RFC-102)', () => {
    const topics: Topic[] = [
      { id: 'topic:z', label: 'zulu', cluster_id: null, cluster_label: null, cluster_size: 0 },
      { id: 'topic:ai', label: 'ai', cluster_id: 'tc:ml', cluster_label: 'machine learning', cluster_size: 5 },
      { id: 'topic:ml', label: 'ml', cluster_id: 'tc:ml', cluster_label: 'machine learning', cluster_size: 5 },
    ]
    const w = mountPanel({ topics, persons: [] })
    // Dominant-cluster label surfaces as the "Theme" lead-in.
    expect(w.text()).toContain('machine learning')
    // Dominant-cluster topics lead (ai, ml), the singleton (zulu) trails.
    const chips = w.findAll('button').filter((b) => ['ai', 'ml', 'zulu'].includes(b.text()))
    expect(chips.map((c) => c.text())).toEqual(['ai', 'ml', 'zulu'])
    // Dominant chips carry the standout ring; the singleton does not.
    expect(chips[0].classes()).toContain('ring-topic')
    expect(chips[2].classes()).not.toContain('ring-topic')
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

  it('hides the insight save-to-highlights control when signed out', () => {
    const w = mountPanel()
    expect(w.find('[aria-label="Save to highlights"]').exists()).toBe(false)
  })

  it('lets a signed-in user save an insight to highlights (P2 capture)', async () => {
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    vi.spyOn(api, 'getHighlights').mockResolvedValue([])
    vi.spyOn(api, 'getNotes').mockResolvedValue([])
    const created: Highlight = {
      id: 'h1', episode_slug: 's1', kind: 'insight', start_ms: 12000, end_ms: null,
      char_start: null, char_end: null, segment_ids: [], quote_text: 'Sleep consolidates memory.',
      speaker: null, source_insight_id: 'i1', color: null, created_at: 1, anchor_status: null,
    }
    const create = vi.spyOn(api, 'createHighlight').mockResolvedValue(created)
    const w = mountPanel()
    await flushPromises()
    const save = w.find('[aria-label="Save to highlights"]')
    expect(save.exists()).toBe(true)
    await save.trigger('click')
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'insight', source_insight_id: 'i1', start_ms: 12000 }),
    )
  })
})
