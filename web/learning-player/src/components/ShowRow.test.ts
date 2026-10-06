import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
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
