import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { resetCorpusLanguagesForTests } from '../composables/useCorpusLanguages'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import type { Podcast } from '../services/types'
import ShowRow from './ShowRow.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
  ],
})

const SHOW: Podcast = {
  feed_id: 'f1',
  title: 'The Show',
  artwork_url: null,
  image_url: null,
  description: 'About the show.',
  episode_count: 3,
}

function mountRow(props: Record<string, unknown> = {}) {
  return mount(ShowRow, {
    props: { show: SHOW, ...props },
    slots: { actions: '<button type="button" data-testid="act">x</button>' },
    global: { plugins: [i18n, router] },
  })
}

describe('ShowRow controls placement', () => {
  it('plates the controls OVER the artwork by default', () => {
    const w = mountRow()
    const act = w.get('[data-testid="act"]')
    expect(act.element.closest('.absolute')).not.toBeNull()
    expect(w.find('[data-testid="show-row-actions"]').exists()).toBe(false)
  })

  it('puts them in a row UNDER the artwork, as the episode card does, with actionsBelow', () => {
    // Library › Saved lists shows and episodes together; one page, one place for controls
    // (operator 2026-10-05). The row is the episode card's: artwork-wide, not plated, in the aside.
    const w = mountRow({ actionsBelow: true })
    const row = w.get('[data-testid="show-row-actions"]')
    expect(row.find('[data-testid="act"]').exists()).toBe(true)
    expect(row.classes()).toEqual(expect.arrayContaining(['w-32', 'gap-[12px]']))
    expect(row.element.closest('.lp-media-aside')).not.toBeNull()
    expect(w.get('[data-testid="act"]').element.closest('.absolute')).toBeNull()
    expect(w.findAll('[data-testid="act"]')).toHaveLength(1) // moved, not duplicated
  })
})

describe('ShowRow saved colour', () => {
  it('draws the same left bar as an episode when the show carries a saved colour', () => {
    const row = mountRow({ color: 'rose' }).get('[data-testid="show-row"]')
    expect(row.classes()).toContain('border-l-4')
    expect(row.classes()).toContain('border-l-rose-400')
  })

  it('draws no bar without one', () => {
    expect(mountRow().get('[data-testid="show-row"]').classes()).not.toContain('border-l-4')
  })
})

describe('ShowRow facts line (operator 2026-10-10)', () => {
  it('leads with the language badge, then the episode count, as on the show page', async () => {
    resetCorpusLanguagesForTests()
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'f1', language: 'en' },
      { feed_id: 'f2', language: 'es' },
    ] as never)
    const w = mountRow({ show: { ...SHOW, language: 'en' } })
    await flushPromises()
    const facts = w.get('[data-testid="show-row-facts"]')
    const badge = facts.element.firstElementChild as HTMLElement
    expect(badge.textContent?.trim().toUpperCase()).toBe('EN') // uppercased by CSS
    expect(facts.text()).toMatch(/^en\s*3 episodes$/i)
    vi.restoreAllMocks()
    resetCorpusLanguagesForTests()
  })
})

