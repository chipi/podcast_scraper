import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import EpisodeTile from './EpisodeTile.vue'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary } from '../services/types'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

const LONG_TITLE =
  'Harrison Chase of LangChain on Deep Agents, LangSmith, and Earning Trust | NVIDIA AI Podcast Ep. 297'

function episode(over: Partial<EpisodeSummary> = {}): EpisodeSummary {
  return {
    slug: 'ep-1',
    title: LONG_TITLE,
    podcast_title: 'NVIDIA AI Podcast',
    artwork_url: '/art.jpg',
    episode_image_url: null,
    feed_image_url: null,
    publish_date: '2026-09-01',
    duration_seconds: 1800,
    status: 'ready',
    ...over,
  } as EpisodeSummary
}

function tile(over: Partial<EpisodeSummary> = {}) {
  setActivePinia(createPinia())
  return mount(EpisodeTile, {
    props: { episode: episode(over) },
    global: { plugins: [i18n, router, createPinia()] },
  })
}

/**
 * The rail slot used `EpisodeCard compact` — a horizontal card whose text column got ~100px in a
 * 224px slot. A real title wrapped to eight lines, the slot grew to ~800px tall, and the action row
 * (positioned against the card's top-right) floated over the artwork.
 */
describe('EpisodeTile', () => {
  it('stacks: artwork first, then actions, then the text', () => {
    const w = tile()
    const kids = Array.from(w.element.children).map((c) => c.tagName.toLowerCase())
    // Two links (artwork, text) around one action row — the order is the layout.
    expect(kids).toHaveLength(3)
    expect(w.element.children[0].querySelector('img'), 'artwork is not first').not.toBeNull()
  })

  it('the title gets the full width and is clamped, not squeezed into a column', () => {
    // The specific failure: a long title in a narrow column becomes a very tall slot. Clamping keeps
    // rail slots the same height; the full width is what stops three words per line.
    const w = tile()
    const title = w.findAll('span').find((s) => s.text().includes('Harrison Chase'))
    expect(title, 'the title did not render').toBeTruthy()
    expect(title!.classes(), 'the title is not clamped').toContain('line-clamp-3')
    expect(title!.classes(), 'the title is not a full-width block').toContain('block')
  })

  it('the actions are BELOW the artwork, never over it', () => {
    // `ShowTile` overlays one follow button deliberately; two icons over episode art is the
    // crowding this replaces. An absolutely-positioned action row is how that comes back.
    const w = tile()
    const actions = w.element.children[1] as HTMLElement
    expect(actions.className, 'the action row is positioned over the artwork').not.toContain('absolute')
    expect(actions.querySelectorAll('button').length).toBeGreaterThanOrEqual(2)
  })

  it('carries TWO actions — the ones a "listen next" decision needs', () => {
    // Four 44px targets cannot sit at a non-overlapping pitch across 176px, and download /
    // add-to-collection belong where you have already committed to the episode.
    const w = tile()
    expect(w.element.children[1].querySelectorAll('button')).toHaveLength(2)
  })

  it('shows no summary — there is no room for one at this width', () => {
    // A two-line truncated fragment is not a summary, it is the shape of one.
    const w = tile({ ...(episode() as object) } as Partial<EpisodeSummary>)
    expect(w.text()).not.toContain('…')
  })

  it('keeps its shape when the episode has no artwork', () => {
    // Otherwise one artless episode collapses its slot and the rail stops lining up.
    const w = tile({ artwork_url: null })
    const first = w.element.children[0] as HTMLElement
    expect(first.querySelector('img')).toBeNull()
    expect(first.innerHTML).toContain('aspect-square')
  })
})
