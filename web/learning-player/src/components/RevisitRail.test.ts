import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Highlight, ResurfacingItem } from '../services/types'
import RevisitRail from './RevisitRail.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/library', name: 'library', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function hl(over: Partial<Highlight> = {}): Highlight {
  return {
    id: 'h1', episode_slug: 'ep-a', kind: 'moment', start_ms: 65_000, end_ms: null,
    char_start: null, char_end: null, segment_ids: [], quote_text: 'a line worth keeping',
    speaker: 'Nora', source_insight_id: null, color: 'amber', created_at: 1,
    anchor_status: null, ...over,
  }
}

const item = (over: Partial<Highlight> = {}): ResurfacingItem => ({
  highlight: hl(over),
  reflection_prompt: 'What still resonates about this?',
})

function mountRail() {
  return mount(RevisitRail, { global: { plugins: [i18n, router] } })
}

describe('RevisitRail (Home)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.spyOn(api, 'getEpisode').mockResolvedValue({
      slug: 'ep-a', title: 'An Episode', artwork_url: 'http://x/art.png',
    } as never)
  })
  afterEach(() => vi.restoreAllMocks())

  it('shows at most four captures, one per episode', async () => {
    // Six due across three episodes: the rail must SAMPLE the breadth, not replay one episode's
    // session, which is what `select_due`'s episode grouping would otherwise hand it.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: false,
      items: [
        item({ id: 'a1', episode_slug: 'ep-a' }),
        item({ id: 'a2', episode_slug: 'ep-a' }),
        item({ id: 'b1', episode_slug: 'ep-b' }),
        item({ id: 'b2', episode_slug: 'ep-b' }),
        item({ id: 'c1', episode_slug: 'ep-c' }),
        item({ id: 'c2', episode_slug: 'ep-c' }),
      ],
    } as never)
    const w = mountRail()
    await flushPromises()
    const cards = w.findAll('[data-testid="home-revisit-card"]')
    expect(cards, 'the rail showed more than one card per episode').toHaveLength(3)
  })

  it('never renders two cards carrying the same words', async () => {
    // The bug this replaced: filling empty slots from episodes already shown. A line saved as BOTH
    // a moment and a quote is two captures with identical text, and the rail rendered both, which
    // reads as broken (observed in a browser 2026-09-18).
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: false,
      items: [
        item({ id: 'a1', episode_slug: 'ep-a', kind: 'moment', quote_text: 'the same line' }),
        item({ id: 'a2', episode_slug: 'ep-a', kind: 'span', quote_text: 'the same line' }),
      ],
    } as never)
    const w = mountRail()
    await flushPromises()
    expect(w.findAll('[data-testid="home-revisit-card"]')).toHaveLength(1)
  })

  it('renders nothing while resurfacing is paused', async () => {
    // The user said stop asking; a rail is the app asking anyway.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: true, items: [item({ id: 'a1' })],
    } as never)
    const w = mountRail()
    await flushPromises()
    expect(w.find('[data-testid="home-revisit-rail"]').exists()).toBe(false)
  })

  it('marking reviewed drops the card and pulls the next episode in', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: false,
      items: [
        item({ id: 'a1', episode_slug: 'ep-a', quote_text: 'first' }),
        item({ id: 'b1', episode_slug: 'ep-b', quote_text: 'second' }),
        item({ id: 'c1', episode_slug: 'ep-c', quote_text: 'third' }),
        item({ id: 'd1', episode_slug: 'ep-d', quote_text: 'fourth' }),
        item({ id: 'e1', episode_slug: 'ep-e', quote_text: 'fifth' }),
      ],
    } as never)
    const marked = vi.spyOn(api, 'markSurfaced').mockResolvedValue(undefined as never)
    const w = mountRail()
    await flushPromises()
    expect(w.text()).toContain('first')

    await w.findAll('[data-testid="home-revisit-reviewed"]')[0].trigger('click')
    await flushPromises()

    expect(marked).toHaveBeenCalledWith('a1')
    expect(w.text(), 'the answered card stayed on screen').not.toContain('first')
    expect(w.text(), 'the freed slot was not backfilled').toContain('fifth')
    expect(w.findAll('[data-testid="home-revisit-card"]')).toHaveLength(4)
  })

  it('muting drops the card the same way, and retires rather than deletes', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: false,
      items: [
        item({ id: 'a1', episode_slug: 'ep-a', quote_text: 'first' }),
        item({ id: 'b1', episode_slug: 'ep-b', quote_text: 'second' }),
      ],
    } as never)
    const retired = vi.spyOn(api, 'retireHighlight').mockResolvedValue(undefined as never)
    const deleted = vi.spyOn(api, 'deleteHighlight')
    const w = mountRail()
    await flushPromises()

    await w.findAll('[data-testid="home-revisit-mute"]')[0].trigger('click')
    await flushPromises()

    expect(retired).toHaveBeenCalledWith('a1')
    expect(deleted, 'mute destroyed the capture — it must only retire it').not.toHaveBeenCalled()
    expect(w.text()).not.toContain('first')
  })

  it('restores the card when the write fails', async () => {
    // A card that vanished without counting would leave the user believing they had reviewed
    // something they had not.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: false, items: [item({ id: 'a1', quote_text: 'first' })],
    } as never)
    vi.spyOn(api, 'markSurfaced').mockRejectedValue(new Error('offline'))
    const w = mountRail()
    await flushPromises()

    await w.find('[data-testid="home-revisit-reviewed"]').trigger('click')
    await flushPromises()

    expect(w.text(), 'a failed review silently swallowed the card').toContain('first')
  })

  it('links to the Revisit tab focused on that capture, not to the player', async () => {
    // From Home the user is deciding what to DO with a capture, and the outcomes live on the
    // Revisit card. Going to the player would also mark it reviewed on arrival (#35) — deciding
    // for them the one thing they went there to decide.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      paused: false, items: [item({ id: 'a1' })],
    } as never)
    const w = mountRail()
    await flushPromises()
    const href = w.find('[data-testid="home-revisit-card"]').attributes('href')
    expect(href).toContain('tab=revisit')
    expect(href).toContain('focus=a1')
    expect(href, 'the card deep-linked into the player').not.toContain('/episode/')
  })
})
