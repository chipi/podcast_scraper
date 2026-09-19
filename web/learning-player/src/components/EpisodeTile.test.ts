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
  it('stacks: the overlaid action row, then artwork, then the text', () => {
    const w = tile()
    const kids = Array.from<Element>(w.element.children).map((c) => c.tagName.toLowerCase())
    // The action row is absolutely positioned, so it leads in source order but paints over the
    // artwork; the two links (artwork, text) are the flow.
    expect(kids).toHaveLength(3)
    expect(w.element.children[1].querySelector('img'), 'artwork is not the first link').not.toBeNull()
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

  it('overlays the actions on the artwork, width-capped so they wrap instead of spilling', () => {
    // Home's "Recommended for you" is the same shape and overlays; the two grids disagreed about
    // where an episode's controls live (operator 2026-09-17). The cap is load-bearing: an
    // absolutely-positioned row sizes to max-content and will not wrap, so four icons ran off a
    // narrow 2-column phone tile.
    const w = tile()
    const actions = w.get('[data-testid="episode-actions"]')
    expect(actions.classes(), 'the action row is not over the artwork').toContain('absolute')
    expect(actions.classes(), 'the row is uncapped and will not wrap').toContain('max-w-[76px]')
    expect(actions.findAll('button').length).toBeGreaterThanOrEqual(2)
  })

  it('names the SHOW above the episode title', () => {
    // The scan order is context-then-title, consistent with EpisodeCard and the Discover grid.
    const w = tile()
    const text = w.text()
    expect(text.indexOf('NVIDIA AI Podcast')).toBeLessThan(text.indexOf('Harrison Chase'))
  })

  it('shows the full shared action set — same as the list card (count must not change by view)', () => {
    // Grid tile and list card render the identical EpisodeActions set, so the action count never
    // changes with the view (operator 2026-09-13). The row is favourite + queue + ⋯ overflow =
    // 3 top-level controls, uniform across web AND native (download + collect live inside the ⋯).
    const w = tile()
    expect(w.get('[data-testid="episode-actions"]').findAll('button')).toHaveLength(3)
  })

  it('shows no summary — there is no room for one at this width', () => {
    // A two-line truncated fragment is not a summary, it is the shape of one.
    const w = tile({ ...(episode() as object) } as Partial<EpisodeSummary>)
    expect(w.text()).not.toContain('…')
  })

  it('keeps its shape when the episode has no artwork', () => {
    // Otherwise one artless episode collapses its slot and the rail stops lining up.
    const w = tile({ artwork_url: null })
    // children[0] is the overlaid action row; the artwork link is the first element in FLOW.
    const artworkLink = w.element.children[1] as HTMLElement
    expect(artworkLink.querySelector('img')).toBeNull()
    expect(artworkLink.innerHTML).toContain('aspect-square')
  })
})
