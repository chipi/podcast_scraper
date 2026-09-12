import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import EpisodeRecapPanel from './EpisodeRecapPanel.vue'
import en from '../i18n/locales/en.json'
import type { EpisodeRecap } from '../services/types'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/topic/:id', name: 'topic', component: { template: '<div/>' } },
    { path: '/storyline/:id', name: 'storyline', component: { template: '<div/>' } },
  ],
})

function recap(over: Partial<EpisodeRecap> = {}): EpisodeRecap {
  return {
    slug: 'ep-1',
    title: 'What we learned about agents',
    podcast_title: 'The Show',
    artwork_url: '/art.jpg',
    key_points: ['Agents need memory', 'Tools beat prompts'],
    summary_text: 'A prose summary of the episode.',
    insights: [
      { id: 'i1', text: 'Grounding matters more than model size.', grounded: true, insight_type: null, confidence: null, position_hint: null, quotes: [] },
    ],
    signature_quote: { text: 'The graph is the product.', speaker: 'Jane Doe', char_start: null, char_end: null, start_ms: null, end_ms: null },
    topics: [
      { id: 'topic:scaling', label: 'Scaling', cluster_id: null, cluster_label: null, cluster_size: 0 },
      { id: 'topic:rag', label: 'Retrieval', cluster_id: null, cluster_label: null, cluster_size: 0 },
    ],
    storylines: [{ id: 'topic:reliability', label: 'The agent-reliability thread' }],
    has_gi: true,
    ...over,
  }
}

interface PanelProps {
  autoAdvanceSeconds?: number | null
  nextTitle?: string | null
}

function panel(recapOver: Partial<EpisodeRecap> = {}, extra: PanelProps = {}) {
  setActivePinia(createPinia())
  return mount(EpisodeRecapPanel, {
    props: { recap: recap(recapOver), ...extra },
    global: { plugins: [i18n, router, createPinia()] },
  })
}

describe('EpisodeRecapPanel', () => {
  it('leads with the finish + the reassurance, and the episode title', () => {
    const w = panel()
    expect(w.text()).toContain('You just finished')
    expect(w.text()).toContain('We took notes for you')
    expect(w.text()).toContain('What we learned about agents')
  })

  it('renders the summary key points', () => {
    const w = panel()
    const points = w.get('[data-testid="recap-key-points"]')
    expect(points.text()).toContain('Agents need memory')
    expect(points.text()).toContain('Tools beat prompts')
  })

  it('falls back to the prose summary when there are no bullets', () => {
    const w = panel({ key_points: [] })
    expect(w.get('[data-testid="recap-key-points"]').text()).toContain('A prose summary of the episode.')
  })

  it('renders the signature quote WITH attribution', () => {
    const q = panel().get('[data-testid="recap-quote"]')
    expect(q.text()).toContain('The graph is the product.')
    expect(q.text()).toContain('Jane Doe')
  })

  it('shows the quote but no attribution line when the speaker is unnamed', () => {
    // #1978: an unnamed voice yields no attribution — the panel never invents one.
    const w = panel({
      signature_quote: { text: 'Anonymous take.', speaker: null, char_start: null, char_end: null, start_ms: null, end_ms: null },
    })
    const q = w.get('[data-testid="recap-quote"]')
    expect(q.text()).toContain('Anonymous take.')
    expect(q.find('footer').exists()).toBe(false)
  })

  it('omits the quote block entirely when there is no signature quote', () => {
    const w = panel({ signature_quote: null })
    expect(w.find('[data-testid="recap-quote"]').exists()).toBe(false)
  })

  it('lists the top insights', () => {
    const w = panel()
    expect(w.get('[data-testid="recap-insights"]').text()).toContain('Grounding matters more than model size.')
  })

  it('renders key-topic chips linking to the topic route', () => {
    const w = panel()
    const topics = w.get('[data-testid="recap-topics"]')
    const links = topics.findAllComponents({ name: 'RouterLink' })
    expect(links).toHaveLength(2)
    expect(links[0].props('to')).toEqual({ name: 'topic', params: { id: 'topic:scaling' } })
    expect(topics.text()).toContain('Scaling')
  })

  it('renders storyline links to the storyline route (by anchor topic id)', () => {
    const w = panel()
    const sl = w.get('[data-testid="recap-storylines"]')
    const link = sl.findComponent({ name: 'RouterLink' })
    expect(link.props('to')).toEqual({ name: 'storyline', params: { id: 'topic:reliability' } })
    expect(sl.text()).toContain('The agent-reliability thread')
  })

  it('omits the topics / storylines sections when empty', () => {
    const w = panel({ topics: [], storylines: [] })
    expect(w.find('[data-testid="recap-topics"]').exists()).toBe(false)
    expect(w.find('[data-testid="recap-storylines"]').exists()).toBe(false)
  })

  it('emits dismiss from both the header close and the footer button (no countdown)', async () => {
    const w = panel()
    expect(w.find('[data-testid="recap-countdown"]').exists()).toBe(false)
    await w.get('[data-testid="recap-dismiss"]').trigger('click')
    await w.get('[data-testid="recap-back"]').trigger('click')
    expect(w.emitted('dismiss')).toHaveLength(2)
  })

  describe('end-card countdown', () => {
    beforeEach(() => vi.useFakeTimers())
    afterEach(() => vi.useRealTimers())

    it('counts down and emits advance when it reaches zero', async () => {
      const w = panel({}, { autoAdvanceSeconds: 3, nextTitle: 'The next episode' })
      const line = w.get('[data-testid="recap-countdown"]')
      expect(line.text()).toContain('Up next in 3s')
      expect(line.text()).toContain('The next episode')
      vi.advanceTimersByTime(3000)
      expect(w.emitted('advance')).toHaveLength(1)
    })

    it('the progress bar depletes with the countdown', async () => {
      const w = panel({}, { autoAdvanceSeconds: 8 })
      const bar = () => w.get('[data-testid="recap-progress"]').attributes('style') ?? ''
      expect(bar()).toContain('width: 100%')
      vi.advanceTimersByTime(4000) // half elapsed
      await w.vm.$nextTick()
      expect(bar()).toContain('width: 50%')
    })

    it('"Play next" advances immediately', async () => {
      const w = panel({}, { autoAdvanceSeconds: 8 })
      await w.get('[data-testid="recap-play-next"]').trigger('click')
      expect(w.emitted('advance')).toHaveLength(1)
      vi.advanceTimersByTime(8000) // the timer was cleared — no second advance
      expect(w.emitted('advance')).toHaveLength(1)
    })

    it('"Stay" cancels the countdown and collapses to Back to player', async () => {
      const w = panel({}, { autoAdvanceSeconds: 8 })
      await w.get('[data-testid="recap-stay"]').trigger('click')
      expect(w.emitted('stay')).toHaveLength(1)
      vi.advanceTimersByTime(10_000)
      expect(w.emitted('advance')).toBeUndefined()
      expect(w.find('[data-testid="recap-countdown"]').exists()).toBe(false)
      expect(w.find('[data-testid="recap-back"]').exists()).toBe(true)
    })

    it('no countdown when autoAdvanceSeconds is null (recap-then-stop)', () => {
      const w = panel({}, { autoAdvanceSeconds: null })
      expect(w.find('[data-testid="recap-countdown"]').exists()).toBe(false)
      expect(w.find('[data-testid="recap-back"]').exists()).toBe(true)
    })
  })
})
