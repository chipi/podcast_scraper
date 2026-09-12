import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import EpisodeRecapPanel from './EpisodeRecapPanel.vue'
import en from '../i18n/locales/en.json'
import type { EpisodeRecap, EpisodeSummary } from '../services/types'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
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
    has_gi: true,
    ...over,
  }
}

function relatedEp(slug: string): EpisodeSummary {
  return {
    slug,
    title: `Peer ${slug}`,
    podcast_title: 'The Show',
    artwork_url: '/p.jpg',
    episode_image_url: null,
    feed_image_url: null,
    publish_date: null,
    duration_seconds: null,
    status: 'ready',
  } as EpisodeSummary
}

function panel(recapOver: Partial<EpisodeRecap> = {}, related: EpisodeSummary[] = []) {
  setActivePinia(createPinia())
  return mount(EpisodeRecapPanel, {
    props: { recap: recap(recapOver), related },
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

  it('shows the "more like this" grid only when related episodes exist', () => {
    expect(panel({}, []).find('[data-testid="recap-more-like-this"]').exists()).toBe(false)
    const w = panel({}, [relatedEp('a'), relatedEp('b')])
    const grid = w.get('[data-testid="recap-more-like-this"]')
    expect(grid.text()).toContain('Listen more like this')
    expect(grid.findAll('li')).toHaveLength(2)
  })

  it('emits dismiss from both the header close and the footer button', async () => {
    const w = panel()
    await w.get('[data-testid="recap-dismiss"]').trigger('click')
    await w.get('[data-testid="recap-back"]').trigger('click')
    expect(w.emitted('dismiss')).toHaveLength(2)
  })
})
