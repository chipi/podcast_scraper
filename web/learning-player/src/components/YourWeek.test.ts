import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import { useUserPreferencesStore } from '../stores/userPreferences'
import type { YourWeekResponse } from '../services/types'
import YourWeek from './YourWeek.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/episode/:slug', name: 'player', component: { template: '<div/>' } }],
})

const EMPTY: YourWeekResponse = { sections: [], period_label: '', generated_at: '' }
const RESP: YourWeekResponse = {
  sections: [
    {
      kind: 'revisit',
      items: [
        {
          episode_slug: 'ep-a',
          episode_title: 'Episode A',
          deep_link: '/episode/ep-a?t=10',
          quote: 'A memorable line.',
          t_ms: 10000,
          graph_refs: [{ id: 'topic:x', kind: 'topic', label: 'Topic X' }],
        },
      ],
    },
    {
      kind: 'new_in_follows',
      items: [
        {
          episode_slug: 'ep-b',
          episode_title: 'Episode B',
          deep_link: '/episode/ep-b',
          graph_refs: [{ id: 'topic:y', kind: 'topic', label: 'Topic Y' }],
        },
      ],
    },
  ],
  period_label: 'Aug 1 – 7',
  generated_at: '2026-08-07T00:00:00Z',
}

function mountIt(opts: { signedIn?: boolean; resp?: YourWeekResponse; layout?: 'full' | 'compact' } = {}) {
  setActivePinia(createPinia())
  const prefs = useUserPreferencesStore()
  vi.spyOn(prefs, 'hydrate').mockResolvedValue()
  vi.spyOn(prefs, 'get').mockReturnValue(opts.layout)
  const setSpy = vi.spyOn(prefs, 'set').mockResolvedValue()
  if (opts.signedIn) {
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  }
  vi.spyOn(api, 'getYourWeek').mockResolvedValue(opts.resp ?? EMPTY)
  const wrapper = mount(YourWeek, { global: { plugins: [i18n, router] } })
  return { wrapper, setSpy }
}

afterEach(() => vi.restoreAllMocks())

describe('YourWeek section', () => {
  it('is hidden when signed out (never calls the API)', async () => {
    const spy = vi.spyOn(api, 'getYourWeek')
    const { wrapper } = mountIt({ signedIn: false, resp: RESP })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false)
    expect(spy).not.toHaveBeenCalled()
  })

  it('is hidden when signed in but nothing is due', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: EMPTY })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false)
  })

  it('renders the compact rail by default with content', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    expect(wrapper.text()).toContain(en.home.yourWeek)
    expect(wrapper.text()).toContain('Episode A')
    expect(wrapper.text()).toContain('A memorable line.')
    expect(wrapper.text()).toContain(en.home.yourWeekShowMore)
    // section labels appear only in the full layout
    expect(wrapper.text()).not.toContain(en.home.yourWeekSection.new_in_follows)
  })

  it('respects a saved full layout and shows per-section labels', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP, layout: 'full' })
    await flushPromises()
    expect(wrapper.text()).toContain(en.home.yourWeekSection.revisit)
    expect(wrapper.text()).toContain(en.home.yourWeekSection.new_in_follows)
    expect(wrapper.text()).toContain(en.home.yourWeekShowLess)
  })

  it('toggles layout inline and persists the preference', async () => {
    const { wrapper, setSpy } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    await wrapper.get('[data-testid="yourweek-toggle"]').trigger('click')
    expect(setSpy).toHaveBeenCalledWith('lp.yourweek.layout', 'full')
    expect(wrapper.text()).toContain(en.home.yourWeekSection.new_in_follows)
  })
})
