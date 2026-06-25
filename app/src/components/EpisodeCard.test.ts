import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createRouter, createMemoryHistory } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary } from '../services/types'
import EpisodeCard from './EpisodeCard.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

beforeEach(() => {
  // Fresh pinia per test; auth defaults to signed-out → no queue button.
  setActivePinia(createPinia())
})
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function makeEpisode(over: Partial<EpisodeSummary> = {}): EpisodeSummary {
  return {
    slug: 'show-abc123',
    title: 'A Great Episode',
    feed_id: 'show',
    podcast_title: 'The Show',
    publish_date: '2024-03-10',
    duration_seconds: 2880,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    status: 'ready',
    summary_preview: 'A crisp recap.',
    summary_bullets: ['Sleep clears metabolic waste.', 'Deep sleep consolidates memory.'],
    topics: ['memory', 'sleep'],
    has_transcript: true,
    has_summary: true,
    has_gi: true,
    has_kg: true,
    has_bridge: false,
    ...over,
  }
}

function mountCard(ep: EpisodeSummary) {
  return mount(EpisodeCard, { props: { episode: ep }, global: { plugins: [i18n, router] } })
}

describe('EpisodeCard', () => {
  it('renders title, podcast, clean lede, duration, and the full-summary insights popover', () => {
    const w = mountCard(makeEpisode())
    expect(w.text()).toContain('A Great Episode')
    expect(w.text()).toContain('The Show')
    expect(w.text()).toContain('A crisp recap.') // clean one-line lede (not the bullets jammed)
    expect(w.text()).toContain('48 min')
    // The insights affordance exposes the FULL summary bullets (popover content is in the DOM).
    expect(w.find('[role="dialog"]').exists()).toBe(true)
    expect(w.text()).toContain('Deep sleep consolidates memory.')
  })

  it('links to the player and to the podcast view', () => {
    const w = mountCard(makeEpisode())
    const hrefs = w.findAll('a').map((a) => a.attributes('href'))
    expect(hrefs).toContain('/episode/show-abc123')
    expect(hrefs).toContain('/podcast/show')
  })

  it('degrades cleanly when enrichment is absent', () => {
    const w = mountCard(
      makeEpisode({
        summary_preview: null,
        summary_bullets: [],
        topics: [],
        has_gi: false,
        duration_seconds: null,
        podcast_title: null,
      }),
    )
    expect(w.text()).toContain('A Great Episode')
    // No insights affordance without grounded summary bullets.
    expect(w.find('[role="dialog"]').exists()).toBe(false)
    expect(w.text()).not.toContain('min')
    // No podcast link when the title is absent.
    expect(w.findAll('a').map((a) => a.attributes('href'))).not.toContain('/podcast/show')
  })

  it('shows pending status when not ready', () => {
    const w = mountCard(makeEpisode({ status: 'pending' }))
    expect(w.text()).toContain('Pending')
  })

  it('prefers local artwork_url over the remote image URLs', () => {
    const w = mountCard(
      makeEpisode({
        artwork_url: '/api/app/artwork?ref=x&size=thumb',
        episode_image_url: 'https://remote/ep.jpg',
        feed_image_url: 'https://remote/feed.jpg',
      }),
    )
    expect(w.find('img').attributes('src')).toBe('/api/app/artwork?ref=x&size=thumb')
  })

  it('falls back to the remote image URL when no local artwork', () => {
    const w = mountCard(makeEpisode({ artwork_url: null, feed_image_url: 'https://remote/feed.jpg' }))
    expect(w.find('img').attributes('src')).toBe('https://remote/feed.jpg')
  })
})
