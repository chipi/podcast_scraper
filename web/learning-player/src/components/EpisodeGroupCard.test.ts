import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as native from '../services/native'
import { flushPromises } from '@vue/test-utils'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import EpisodeGroupCard from './EpisodeGroupCard.vue'
import en from '../i18n/locales/en.json'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})
const episode = {
  slug: 'ep-1',
  title: 'An Episode',
  podcast_title: 'A Show',
  artwork_url: '/a.jpg',
  episode_image_url: null,
  feed_image_url: null,
}

function card(props: Record<string, unknown> = {}) {
  setActivePinia(createPinia())
  return mount(EpisodeGroupCard, {
    props: { episode, itemCount: 2, ...props },
    slots: { default: '<p data-testid="row">row</p>', meta: 'Oct 1 · 2 items' },
    global: { plugins: [i18n, router, createPinia()] },
  })
}

/** One header for Search, Saved and Revisit (operator 2026-10-05). */
describe('EpisodeGroupCard', () => {
  it('shows the show ABOVE the title, and one meta line', () => {
    const w = card()
    const text = w.get('[data-testid="episode-group-link"]').text()
    expect(text.indexOf('A Show')).toBeLessThan(text.indexOf('An Episode'))
    expect(w.get('[data-testid="episode-group-meta"]').text()).toBe('Oct 1 · 2 items')
  })

  it('folds with a chevron and keeps the body in the DOM', async () => {
    const w = card()
    const toggle = w.get('[data-testid="episode-group-toggle"]')
    await toggle.trigger('click')
    expect(w.emitted('update:expanded')?.[0]).toEqual([false])
    expect(w.get('[data-testid="episode-group-body"]').attributes('style')).toContain('display: none')
    expect(w.find('[data-testid="row"]').exists()).toBe(true)
  })

  it('follows a view-owned expanded state', () => {
    const w = card({ expanded: false })
    expect(w.get('[data-testid="episode-group-toggle"]').attributes('aria-expanded')).toBe('false')
  })

  it('has no fold control when there is nothing to fold', () => {
    expect(card({ itemCount: 0 }).find('[data-testid="episode-group-toggle"]').exists()).toBe(false)
  })

  it('the chevron sits on the bottom row beside the count it folds (operator 2026-10-08)', () => {
    const w = card()
    const meta = w.get('[data-testid="episode-group-meta"]')
    expect(meta.element.parentElement?.contains(w.get('[data-testid="episode-group-toggle"]').element)).toBe(true)
  })

  it('the ⋯ and the chevron are siblings of the link, not inside it', () => {
    const link = card().get('[data-testid="episode-group-link"]')
    expect(link.findAll('button')).toHaveLength(0)
  })

  it('its ⋯ offers Share, sending the episode link to the share sheet (operator 2026-10-08)', async () => {
    const sheet = vi.spyOn(native, 'openShareSheet').mockResolvedValue()
    setActivePinia(createPinia())
    const w = mount(EpisodeGroupCard, {
      props: { episode, itemCount: 2 },
      global: { plugins: [i18n, router, createPinia()], stubs: { teleport: true } },
    })
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    await w.get('[data-testid="episode-share"]').trigger('click')
    await flushPromises()
    expect(sheet).toHaveBeenCalledOnce()
    const [title, url] = sheet.mock.calls[0]
    expect(title).toBe('An Episode')
    expect(url).toMatch(/\/episode\/ep-1$/)
    vi.restoreAllMocks()
  })
})

describe('EpisodeGroupCard and Moments (operator 2026-10-10)', () => {
  // Online, so the link is live (offline it greys; MomentsEntry.test.ts covers that).
  beforeEach(async () => {
    const { reportServerReachable } = await import('../composables/useOnline')
    reportServerReachable(true)
  })
  it('Search: "▶ Moments" on the meta row when asked for the link', () => {
    expect(card({ moments: true, momentsLink: true }).find('[data-testid="moments-link"]').exists()).toBe(true)
  })
  it('Saved / Revisit: the reel is in the ⋯ menu only, no link on the row', () => {
    expect(card({ moments: true }).find('[data-testid="moments-link"]').exists()).toBe(false)
  })
  it('no moments, no link even when asked', () => {
    expect(card({ momentsLink: true }).find('[data-testid="moments-link"]').exists()).toBe(false)
  })
})
