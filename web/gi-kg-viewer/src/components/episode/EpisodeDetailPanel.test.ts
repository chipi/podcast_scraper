// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h } from 'vue'

import EpisodeDetailPanel from './EpisodeDetailPanel.vue'
import { episodeDetailSections } from '../../extensions'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { CorpusEpisodeDetailResponse } from '../../api/corpusLibraryApi'

// ── API mocks ────────────────────────────────────────────────────────────────
// The panel is driven by the subject store (focused episode) and fetches feeds,
// index stats, episode detail, similar episodes, and related insights. Mock all
// network modules so mount tests exercise the render states + interactions.

const fetchCorpusFeeds = vi.fn()
const fetchCorpusEpisodeDetail = vi.fn()
const fetchCorpusSimilarEpisodes = vi.fn()
vi.mock('../../api/corpusLibraryApi', () => ({
  fetchCorpusFeeds: (...a: unknown[]) => fetchCorpusFeeds(...a),
  fetchCorpusEpisodeDetail: (...a: unknown[]) => fetchCorpusEpisodeDetail(...a),
  fetchCorpusSimilarEpisodes: (...a: unknown[]) => fetchCorpusSimilarEpisodes(...a),
}))

const fetchIndexStats = vi.fn()
vi.mock('../../api/indexStatsApi', () => ({
  fetchIndexStats: (...a: unknown[]) => fetchIndexStats(...a),
}))

// ── Child stubs ──────────────────────────────────────────────────────────────
// PodcastCover (binary art fetch) + HelpTip (popover) + DiagnosticRow +
// CilTopicPillsRow + EpisodeBridgePartition (own tests) stubbed to passthroughs.
const STUBS = {
  PodcastCover: true,
  DiagnosticRow: true,
  EpisodeBridgePartition: true,
  HelpTip: { name: 'HelpTip', template: '<div data-stub="help-tip"><slot /></div>' },
  CilTopicPillsRow: {
    name: 'CilTopicPillsRow',
    props: ['pills'],
    emits: ['pill-click'],
    template:
      '<div data-stub="cil-pills"><button data-testid="cil-pill" @click="$emit(\'pill-click\', 0)" /></div>',
  },
}

function detailOf(over: Partial<CorpusEpisodeDetailResponse> = {}): CorpusEpisodeDetailResponse {
  return {
    path: '/corpus',
    metadata_relative_path: 'meta/ep1.json',
    feed_id: 'feed-a',
    episode_id: 'ep-1',
    episode_title: 'The First Episode',
    publish_date: '2026-01-02',
    summary_title: 'A good summary',
    summary_bullets: ['Point one', 'Point two'],
    summary_text: 'Full summary body text.',
    gi_relative_path: 'a/ep1.gi.json',
    kg_relative_path: 'a/ep1.kg.json',
    has_gi: true,
    has_kg: true,
    cil_digest_topics: [],
    bridge_partition: null,
    ...over,
  }
}

function similarOk(items: unknown[] = []) {
  return {
    path: '/corpus',
    source_metadata_relative_path: 'meta/ep1.json',
    query_used: 'embedded query string',
    items,
    error: null,
    detail: null,
  }
}

/** Mount the panel and focus an episode, then flush the async load chain. */
async function mountWithEpisode(
  detail: CorpusEpisodeDetailResponse | null,
  props: Record<string, unknown> = {},
) {
  const w = mount(EpisodeDetailPanel, { props, attachTo: document.body, global: { stubs: STUBS } })
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const subject = useSubjectStore()
  if (detail) {
    fetchCorpusEpisodeDetail.mockResolvedValue(detail)
    subject.focusEpisode(detail.metadata_relative_path, { uiTitle: detail.episode_title })
  }
  // Flush: feeds → index → detail → similar → related (each awaited microtasks).
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return { w, shell, subject }
}

describe('EpisodeDetailPanel.vue', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    fetchCorpusFeeds.mockResolvedValue({ path: '/corpus', feeds: [] })
    fetchIndexStats.mockResolvedValue({ available: true, stats: null })
    fetchCorpusSimilarEpisodes.mockResolvedValue(similarOk([]))
  })

  it('renders the "No episode selected" empty state with no focused episode', () => {
    const w = mount(EpisodeDetailPanel, { attachTo: document.body, global: { stubs: STUBS } })
    expect(w.text()).toContain('No episode selected.')
  })

  it('renders the episode title + summary once detail loads', async () => {
    const { w } = await mountWithEpisode(detailOf())
    expect(w.find('.node-detail-primary-title').text()).toBe('The First Episode')
    expect(w.text()).toContain('A good summary')
    expect(w.text()).toContain('Full summary body text.')
    expect(w.text()).toContain('Point one')
    expect(w.text()).toContain('Point two')
  })

  it('shows the detail error state when fetchCorpusEpisodeDetail rejects', async () => {
    fetchCorpusEpisodeDetail.mockRejectedValue(new Error('boom detail'))
    const w = mount(EpisodeDetailPanel, { attachTo: document.body, global: { stubs: STUBS } })
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    useSubjectStore().focusEpisode('meta/ep1.json', { uiTitle: 'X' })
    for (let i = 0; i < 8; i++) await w.vm.$nextTick()
    expect(w.find('.text-danger').text()).toContain('boom detail')
  })

  it('renders the "No summary text" fallback when summary is empty', async () => {
    const { w } = await mountWithEpisode(
      detailOf({ summary_title: null, summary_text: null, summary_bullets: [] }),
    )
    expect(w.text()).toContain('No summary text in metadata.')
  })

  it('disables "Open in graph" only when the episode has no GI/KG artifacts', async () => {
    const { w } = await mountWithEpisode(detailOf({ has_gi: false, has_kg: false }))
    const openBtn = w
      .findAll('button')
      .find((b) => b.text().includes('Open in graph'))!
    expect(openBtn.attributes('disabled')).toBeDefined()
  })

  it('emits focus-search with feed + query from "Prefill semantic search"', async () => {
    const { w } = await mountWithEpisode(detailOf({ feed_id: 'feed-a' }))
    const btn = w.findAll('button').find((b) => b.text().includes('Prefill semantic search'))!
    await btn.trigger('click')
    const ev = w.emitted('focus-search')
    expect(ev).toHaveLength(1)
    const payload = ev![0][0] as { feed: string; query: string }
    expect(payload.feed).toBe('feed-a')
    expect(typeof payload.query).toBe('string')
  })

  it('renders the similar-episodes empty state when the search runs with no matches', async () => {
    const { w } = await mountWithEpisode(detailOf())
    expect(w.find('[data-testid="library-similar-empty"]').exists()).toBe(true)
  })

  it('renders similar episode rows and focuses one on click', async () => {
    fetchCorpusSimilarEpisodes.mockResolvedValue(
      similarOk([
        {
          score: 0.912,
          feed_id: 'feed-a',
          episode_id: 'ep-2',
          episode_title: 'A Similar Episode',
          metadata_relative_path: 'meta/ep2.json',
          publish_date: '2026-02-01',
          doc_type: null,
          snippet: '',
        },
      ]),
    )
    const { w, subject } = await mountWithEpisode(detailOf())
    expect(w.find('[data-testid="library-similar"]').text()).toContain('A Similar Episode')
    const row = w.find('[data-testid="library-similar"]').findAll('button').at(-1)!
    await row.trigger('click')
    expect(subject.kind).toBe('episode')
    expect(subject.episodeMetadataPath).toBe('meta/ep2.json')
  })

  it('maps the no_index similar error to an operator-friendly message', async () => {
    fetchCorpusSimilarEpisodes.mockResolvedValue({
      ...similarOk([]),
      error: 'no_index',
    })
    const { w } = await mountWithEpisode(detailOf())
    expect(w.find('[data-testid="library-similar"]').text()).toContain(
      'No vector index for this corpus yet',
    )
  })

  // ── Rail tab branches (prop-driven) ─────────────────────────────────────────

  it('renders the neighbourhood tab slot when railNeighbourhoodEnabled + tab=neighbourhood', async () => {
    const { w } = await mountWithEpisode(detailOf(), {
      railNeighbourhoodEnabled: true,
      railDetailTab: 'neighbourhood',
    })
    expect(w.find('#episode-detail-rail-panel-neighbourhood').exists()).toBe(true)
    const details = w.find('#episode-detail-rail-panel-details')
    expect(details.attributes('role')).toBe('tabpanel')
  })

  it('keeps the details panel mounted (v-show) on the details rail tab', async () => {
    const { w } = await mountWithEpisode(detailOf(), {
      railNeighbourhoodEnabled: true,
      railDetailTab: 'details',
    })
    expect(w.find('#episode-detail-rail-panel-details').exists()).toBe(true)
    expect(w.find('#episode-detail-rail-panel-neighbourhood').exists()).toBe(false)
  })

  it('renders the enrichment panel as a tabpanel on the enrichment rail tab', async () => {
    const { w } = await mountWithEpisode(detailOf(), {
      railNeighbourhoodEnabled: true,
      railDetailTab: 'enrichment',
    })
    const enrichment = w.find('#episode-detail-rail-panel-enrichment')
    expect(enrichment.exists()).toBe(true)
    expect(enrichment.attributes('role')).toBe('tabpanel')
    expect(enrichment.attributes('aria-labelledby')).toBe('episode-detail-rail-tab-enrichment')
    // Details stays mounted (v-show) but hidden; enrichment section lives only here.
    expect(w.find('#episode-detail-rail-panel-details').exists()).toBe(true)
    expect(w.find('#episode-detail-rail-panel-neighbourhood').exists()).toBe(false)
  })

  describe('episodeDetailSections', () => {
    afterEach(() => episodeDetailSections.reset())

    function section(region: 'summary-head' | 'end' | 'enrichment') {
      return defineComponent({
        props: { detail: { type: Object, required: true }, corpusPath: String },
        setup(p) {
          return () =>
            h('div', { 'data-testid': `ext-${region}` }, `${p.detail.episode_id}@${p.corpusPath}`)
        },
      })
    }

    it('renders a registered section in each region with the episode and the corpus', async () => {
      for (const region of ['summary-head', 'end', 'enrichment'] as const) {
        episodeDetailSections.register({ id: region, region, component: section(region) })
      }
      const { w } = await mountWithEpisode(detailOf(), { railDetailTab: 'details' })
      const details = w.find('#episode-detail-rail-panel-details')
      expect(details.find('[data-testid="ext-summary-head"]').text()).toBe('ep-1@/corpus')
      expect(details.find('[data-testid="ext-end"]').text()).toBe('ep-1@/corpus')
      expect(w.find('#episode-detail-rail-panel-enrichment [data-testid="ext-enrichment"]').text()).toBe(
        'ep-1@/corpus',
      )
    })

    it('passes a section’s switch-main-tab and an Enrichment section’s has-content up', async () => {
      episodeDetailSections.register({
        id: 'head',
        region: 'summary-head',
        component: defineComponent({
          props: ['detail', 'corpusPath'],
          emits: ['switch-main-tab'],
          setup(_p, { emit }) {
            return () =>
              h('button', { 'data-testid': 'ext-go', onClick: () => emit('switch-main-tab', 'graph') })
          },
        }),
      })
      episodeDetailSections.register({
        id: 'enr',
        region: 'enrichment',
        component: defineComponent({
          props: ['detail', 'corpusPath'],
          emits: ['has-content'],
          setup(_p, { emit }) {
            emit('has-content', true)
            return () => h('div')
          },
        }),
      })
      const { w } = await mountWithEpisode(detailOf())
      await w.get('[data-testid="ext-go"]').trigger('click')
      expect(w.emitted('switch-main-tab')?.[0]).toEqual(['graph'])
      expect(w.emitted('enrichment-has-content')?.[0]).toEqual([true])
    })

    it('renders nothing extra when nothing is registered', async () => {
      const { w } = await mountWithEpisode(detailOf())
      expect(w.find('[data-testid^="ext-"]').exists()).toBe(false)
    })
  })
})
