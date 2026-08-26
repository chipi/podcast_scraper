import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import FollowedInterests from './FollowedInterests.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/topic/:id', name: 'topic', component: stub },
    { path: '/person/:id', name: 'person', component: stub },
  ],
})

async function mountIt() {
  setActivePinia(createPinia())
  await router.push('/')
  await router.isReady()
  const w = mount(FollowedInterests, { global: { plugins: [i18n, router], stubs: { teleport: true } } })
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('FollowedInterests', () => {
  it('groups followed topics / people / storylines and can unfollow', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['topic:ai-safety', 'person:jane-doe', 'thc:energy'])
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([])
    vi.spyOn(api, 'getStorylines').mockResolvedValue([
      { id: 'thc:energy', label: 'Energy transition', size: 3, anchor_topic_id: 'topic:energy' },
    ])
    const remove = vi.spyOn(api, 'removeInterest').mockResolvedValue(['person:jane-doe', 'thc:energy'])

    const w = await mountIt()
    expect(w.text()).toContain('ai safety') // topic de-slugged
    expect(w.text()).toContain('jane doe') // person de-slugged
    expect(w.text()).toContain('Energy transition') // storyline label resolved from getStorylines

    // Unfollow the first item.
    await w.findAll('[data-testid="unfollow"]')[0].trigger('click')
    expect(remove).toHaveBeenCalledWith('topic:ai-safety')
  })

  it('shows an empty message when nothing is followed', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getTopClusters').mockResolvedValue([])
    vi.spyOn(api, 'getStorylines').mockResolvedValue([])
    const w = await mountIt()
    expect(w.text()).toContain("not following any topics, people, or storylines")
  })
})
