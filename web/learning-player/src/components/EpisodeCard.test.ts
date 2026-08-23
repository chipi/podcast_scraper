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
  it('renders title, podcast, clean lede and duration', () => {
    const w = mountCard(makeEpisode())
    expect(w.text()).toContain('A Great Episode')
    expect(w.text()).toContain('The Show')
    expect(w.text()).toContain('A crisp recap.') // clean lede, not the bullets jammed together
    expect(w.text()).toContain('48 min')
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

  // --- insights disclosure (#1583) ---
  //
  // These replace the tests that pinned the whole-card hover overlay and the sparkle popover. Both
  // mechanisms were deleted; see the component docblock for why. The assertions below encode the
  // properties that made them wrong, so a reintroduction fails here.

  it('keeps insights collapsed until asked, and out of the accessibility tree', () => {
    const w = mountCard(makeEpisode())
    const toggle = w.get('[aria-expanded]')
    expect(toggle.attributes('aria-expanded')).toBe('false')
    // NOT merely visually hidden: the old overlay used opacity-0, which leaves text in the a11y
    // tree, so every card in a 20-card list read its entire summary to a screen reader.
    expect(w.text()).not.toContain('Deep sleep consolidates memory.')
  })

  it('expands the bullets in place on click — the same gesture on touch and pointer', async () => {
    const w = mountCard(makeEpisode())
    await w.get('[aria-expanded]').trigger('click')
    expect(w.get('[aria-expanded]').attributes('aria-expanded')).toBe('true')
    expect(w.text()).toContain('Deep sleep consolidates memory.')
  })

  it('collapses again on a second click', async () => {
    const w = mountCard(makeEpisode())
    await w.get('[aria-expanded]').trigger('click')
    await w.get('[aria-expanded]').trigger('click')
    expect(w.get('[aria-expanded]').attributes('aria-expanded')).toBe('false')
    expect(w.text()).not.toContain('Deep sleep consolidates memory.')
  })

  it('never renders summary_text — unbounded prose belongs on the player page', () => {
    // The overlay's core defect: it rendered the FULL summary into a fixed-height, overflow-hidden
    // box, slicing long text mid-sentence with no ellipsis and no scroll.
    const w = mountCard(
      makeEpisode({ summary_text: 'A very long unbounded editorial pull-quote.'.repeat(20) }),
    )
    expect(w.text()).not.toContain('A very long unbounded editorial pull-quote.')
  })

  it('has no hover-triggered reveal anywhere on the card', () => {
    // group-hover is not a gesture on touch, and with no hover intent it strobed every card as the
    // pointer passed down a list.
    expect(mountCard(makeEpisode()).html()).not.toContain('group-hover:opacity')
  })

  it('caps how many bullets a card shows, and links out for the rest', async () => {
    // Sized against production, not the fixtures: real bullets run ~207 chars median with 7.9 per
    // episode (measured over 393 bullets, 2026-08-13), so rendering all of them would put ~1,600
    // characters in a list card — the same "doesn't fit" failure as the old overlay, just opt-in.
    const many = Array.from({ length: 9 }, (_, i) => `Grounded claim number ${i} about the topic.`)
    const w = mountCard(makeEpisode({ summary_bullets: many, has_gi: true }))
    await w.get('[aria-expanded]').trigger('click')

    expect(w.findAll('li').length).toBe(4)
    expect(w.text()).toContain('Grounded claim number 3')
    expect(w.text()).not.toContain('Grounded claim number 4')
    expect(w.text()).toContain('+5 more insights')
  })

  it('does not truncate the bullets it does show', async () => {
    // The old overlay sliced prose mid-sentence. A bullet the user explicitly expanded must be
    // readable end to end — length is bounded by the cap above, not by clamping each claim.
    const long = 'A'.repeat(380) // production max
    const w = mountCard(makeEpisode({ summary_bullets: [long], has_gi: true }))
    await w.get('[aria-expanded]').trigger('click')
    expect(w.get('li span:last-child').classes().join(' ')).not.toContain('line-clamp')
    expect(w.text()).toContain(long)
  })

  it('omits the insights affordance when there are no grounded bullets', () => {
    const w = mountCard(makeEpisode({ summary_bullets: [], has_gi: false }))
    expect(w.find('[aria-expanded]').exists()).toBe(false)
  })
})
