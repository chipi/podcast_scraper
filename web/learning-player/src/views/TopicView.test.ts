import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import TopicView from './TopicView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/topic/:id', name: 'topic', component: TopicView, props: true },
    ],
  })
}

beforeEach(() => {
  vi.spyOn(api, 'getTopicCard').mockResolvedValue({
    id: 'topic:ai',
    label: 'Artificial Intelligence',
    cluster_id: 'tc:ai-safety',
    cluster_label: 'AI safety',
    cluster_size: 12,
    sibling_topics: [],
    episode_count: 3,
    episodes: [],
    related_people: [],
  })
})
afterEach(() => vi.restoreAllMocks())

describe('TopicView (#1261-6)', () => {
  it('fetches the topic card via the route param and renders the topic label', async () => {
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push({ name: 'topic', params: { id: 'topic:ai' } })
    await router.isReady()
    const w = mount(TopicView, {
      props: { id: 'topic:ai' },
      global: { plugins: [i18n, router, createPinia()], stubs: { teleport: true } },
    })
    await flushPromises()
    expect(api.getTopicCard).toHaveBeenCalledWith('topic:ai', undefined)
    expect(w.find('[data-testid="topic-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Artificial Intelligence')
  })
})
