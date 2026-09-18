import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Highlight, ResurfacingItem } from '../services/types'
import ResurfacingInbox from './ResurfacingInbox.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

function hl(over: Partial<Highlight> = {}): Highlight {
  return {
    id: 'h1', episode_slug: 'show-ep01', kind: 'moment', start_ms: 65_000, end_ms: null,
    char_start: null, char_end: null, segment_ids: [], quote_text: null, speaker: null,
    source_insight_id: null, color: null, created_at: 1, anchor_status: null, ...over,
  }
}

const item = (over: Partial<ResurfacingItem> = {}): ResurfacingItem => ({
  highlight: hl(),
  reflection_prompt: 'What still resonates about this?',
  ...over,
})

const mountInbox = () => {
  // The inbox writes through the resurfacing store now (#2004 item 14 follow-up), so it needs one.
  setActivePinia(createPinia())
  return mount(ResurfacingInbox, { global: { plugins: [i18n, router, createPinia()] } })
}

beforeEach(() => {
  vi.spyOn(api, 'markSurfaced').mockResolvedValue()
  vi.spyOn(api, 'putResurfacingSettings').mockResolvedValue({ paused: true })
})
afterEach(() => vi.restoreAllMocks())

describe('ResurfacingInbox', () => {
  it('shows the empty state when nothing is due', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [], paused: false })
    const w = mountInbox()
    await flushPromises()
    expect(w.text()).toContain('Nothing to revisit right now')
  })

  it('renders due items with the reflection prompt + jump link, and dismisses one', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    const w = mountInbox()
    await flushPromises()
    expect(w.text()).toContain('What still resonates about this?')
    // jump link carries ?t=65 (65_000ms) to the player. Addressed by testid, not by "the first
    // link mentioning the episode": the rows are grouped under an episode HEADING that links to the
    // episode top, so that locator matched the heading and saw no timestamp.
    const link = w.find('[data-testid="revisit-jump"]')
    expect(link.attributes('href')).toContain('t=65')
    // dismiss removes it locally + advances the ladder server-side
    // Addressed by testid, not by its text: the control is the card's right-hand action edge now
    // (a ✓ glyph over a short "Reviewed" label), so matching the full sentence found nothing.
    await w.find('[data-testid="revisit-dismiss"]').trigger('click')
    expect(api.markSurfaced).toHaveBeenCalledWith('h1')
    await flushPromises()
    expect(w.text()).not.toContain('What still resonates about this?')
  })

  // --- what advances the spaced ladder, and what must not (#35) ---

  it('does NOT advance the ladder just because the tab was opened', async () => {
    // The conservative half of the contract, locked in. onMounted only GETs /resurfacing; if
    // rendering ever started marking, merely glancing at the Revisit tab would count as reviewing
    // every due item at once and the schedule would collapse on first open.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      items: [item(), item({ highlight: hl({ id: 'h2' }) })],
      paused: false,
    })
    mountInbox()
    await flushPromises()
    expect(api.markSurfaced).not.toHaveBeenCalled()
  })

  it('the jump link carries ?revisit=<id> so arriving at the player advances the ladder', async () => {
    // Following the jump IS reviewing (product call, 2026-08-17) — but it is marked on ARRIVAL,
    // not on click, so a cancelled navigation does not consume a repetition. Before this the only
    // advance path was the dismiss button, so a user who genuinely revisited never progressed and
    // the digest re-sent the same five items indefinitely.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    const w = mountInbox()
    await flushPromises()
    const href = w.find('[data-testid="revisit-jump"]').attributes('href') ?? ''
    expect(href).toContain('revisit=h1')
    expect(href).toContain('t=65')
    expect(api.markSurfaced).not.toHaveBeenCalled() // rendering the link marks nothing
  })

  it('pause toggles the pacing setting and reloads', async () => {
    const get = vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    const w = mountInbox()
    await flushPromises()
    get.mockResolvedValue({ items: [], paused: true }) // server now reports paused
    await w.get('[data-testid="revisit-pause"]').trigger('click')
    await flushPromises()
    expect(api.putResurfacingSettings).toHaveBeenCalledWith(true)
    expect(w.text()).toContain('Resurfacing is paused.')
  })

  // --- what each due item is FROM and ABOUT (operator 2026-09-17) ---

  it('groups due items under the episode they came from, by title', async () => {
    // A flat list said nothing about WHERE a moment was from, so two moments from one episode read
    // as two unrelated cards. Same structure as Saved and Search: episode heading, then its items.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      items: [
        item(),
        item({ highlight: hl({ id: 'h2', start_ms: 5_000 }) }),
        item({ highlight: hl({ id: 'h3', episode_slug: 'show-ep02' }) }),
      ],
      paused: false,
    })
    vi.spyOn(api, 'getEpisode').mockImplementation(
      async (slug: string) =>
        ({ slug, title: slug === 'show-ep01' ? 'Risk as a system' : 'Pacing' }) as never,
    )
    const w = mountInbox()
    await flushPromises()
    const groups = w.findAll('[data-testid="revisit-group"]')
    expect(groups).toHaveLength(2) // two episodes, not three cards
    // The heading is the shared EpisodeRow (artwork + title + show), the same row Saved uses.
    const headings = groups.map((g) => g.get('[data-testid="episode-row"]').text())
    expect(headings[0]).toContain('Risk as a system')
    expect(headings[1]).toContain('Pacing')
    expect(groups[0].findAll('[data-testid="revisit-item"]')).toHaveLength(2)
  })

  it('still renders the group when the episode cannot be resolved', async () => {
    // The list is useful before the episode arrives, and one dead episode must not drop a group:
    // the card falls back to the slug as its title rather than the group vanishing.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new Error('gone'))
    const w = mountInbox()
    await flushPromises()
    const group = w.get('[data-testid="revisit-group"]')
    expect(group.get('[data-testid="episode-row"]').text()).toContain('show-ep01')
    expect(group.findAll('[data-testid="revisit-item"]')).toHaveLength(1)
  })

  it('collapses an episode group and restores it', async () => {
    // Collapsible on BOTH Search and Revisit via the shared EpisodeGroupCard (operator). Groups
    // start OPEN — collapsing is an affordance for a long page, not a new default that hides what
    // the listener came for.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    vi.spyOn(api, 'getEpisode').mockResolvedValue({ slug: 'show-ep01', title: 'Risk' } as never)
    const w = mountInbox()
    await flushPromises()
    // Same fold control as Library -> Saved: it rides EpisodeRow's `#trailing` slot and hides the
    // list with `v-show`, so the moments stay in the DOM and re-opening keeps their state.
    const toggle = w.get('[data-testid="revisit-group-collapse"]')
    expect(toggle.attributes('aria-expanded')).toBe('true')
    const list = () => w.get('[data-testid="revisit-group"]').find('ul')
    expect(list().attributes('style') ?? '').not.toContain('display: none')
    await toggle.trigger('click')
    expect(toggle.attributes('aria-expanded')).toBe('false')
    expect(list().attributes('style')).toContain('display: none')
    await toggle.trigger('click')
    expect(list().attributes('style') ?? '').not.toContain('display: none')
  })

  it('labels a moment KIND · DATE and shows the captured words as the body', async () => {
    // "Marked moment" used to BE the body text, so the card said nothing about itself. It is the
    // label; the quote is the content.
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({
      items: [
        item({
          highlight: hl({ quote_text: 'Correlation is the real exposure', speaker: 'Ada' }),
        }),
      ],
      paused: false,
    })
    vi.spyOn(api, 'getEpisode').mockResolvedValue({ slug: 'show-ep01', title: 'Risk' } as never)
    const w = mountInbox()
    await flushPromises()
    expect(w.get('[data-testid="revisit-item"]').text()).toContain('Marked moment')
    expect(w.get('[data-testid="revisit-quote"]').text()).toBe('Correlation is the real exposure')
    expect(w.get('[data-testid="revisit-item"]').text()).toContain('Ada')
    // The prompt survives the restructure — it is the question asked OF the moment, below it.
    expect(w.get('[data-testid="revisit-prompt"]').text()).toBe('What still resonates about this?')
  })

  it('renders no quote block for a moment stored without text', async () => {
    vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [item()], paused: false })
    vi.spyOn(api, 'getEpisode').mockResolvedValue({ slug: 'show-ep01', title: 'Risk' } as never)
    const w = mountInbox()
    await flushPromises()
    expect(w.find('[data-testid="revisit-quote"]').exists()).toBe(false)
  })
})
