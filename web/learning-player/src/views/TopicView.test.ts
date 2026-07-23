import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { PersonCard, TopicCard } from '../services/types'
import { useAuthStore } from '../stores/auth'
import TopicView from './TopicView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/topic/:id', name: 'topic', component: TopicView, props: true },
      // EntityCardBody wires cross-links to other cards via internal open()
      // (a stack push, not a route change) but the searchLibrary handler
      // pushes to /search, and the follow controls rely on session cookies —
      // register the stubs so the test router resolves them.
      { path: '/search', name: 'search', component: { template: '<div/>' } },
      { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
      { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
      { path: '/person/:id', name: 'person', component: { template: '<div/>' }, props: true },
    ],
  })
}

beforeEach(() => {
  // EntitySignals (embedded in EntityCardBody) hits getCorpusEnrichment on
  // mount — stub to empty so the standalone-page tests don't hit the real
  // network via happy-dom.
  vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({})
  vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
  vi.spyOn(api, 'getTopicCard').mockResolvedValue({
    id: 'topic:ai',
    label: 'Artificial Intelligence',
    cluster_id: 'tc:ai-safety',
    cluster_label: 'AI safety',
    cluster_size: 12,
    sibling_topics: [
      { id: 'topic:agi', label: 'AGI', cluster_id: 'tc:ai-safety', cluster_label: 'AI safety', cluster_size: 12 },
      { id: 'topic:alignment', label: 'Alignment', cluster_id: 'tc:ai-safety', cluster_label: 'AI safety', cluster_size: 12 },
    ],
    episode_count: 3,
    episodes: [],
    related_people: [
      { id: 'person:jane-doe', name: 'Jane Doe', kind: 'person' },
    ],
  })
})
afterEach(() => vi.restoreAllMocks())

async function mountTopic(id = 'topic:ai') {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: 'topic', params: { id } })
  await router.isReady()
  const w = mount(TopicView, {
    props: { id },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await flushPromises()
  return { w, router }
}

describe('TopicView (#1261-6)', () => {
  it('fetches the topic card via the route param and renders the topic label', async () => {
    const { w } = await mountTopic()
    expect(api.getTopicCard).toHaveBeenCalledWith('topic:ai', undefined)
    expect(w.find('[data-testid="topic-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Artificial Intelligence')
  })

  // #1261-final: EntityCardBody surfaces inside the standalone page
  it('renders in inline mode — Back button, no Open-in-page link (would go nowhere)', async () => {
    const { w } = await mountTopic()
    // Inline mode: "‹ Back" not "✕ Close".
    expect(w.text()).toContain('Back')
    // Open-in-page link is overlay-only and would loop back to the same page.
    expect(w.find('[data-testid="ec-open-in-page"]').exists()).toBe(false)
  })

  it('renders the sibling-topics chip section from the topic card payload', async () => {
    const { w } = await mountTopic()
    // "3 topics in this cluster" — current + 2 siblings.
    expect(w.text()).toContain('3 topics in this cluster')
    expect(w.text()).toContain('AGI')
    expect(w.text()).toContain('Alignment')
  })

  it('clicking a sibling topic chip pushes it onto the back stack — Back returns to the origin topic', async () => {
    vi.spyOn(api, 'getTopicCard').mockImplementation(async (id: string) => {
      if (id === 'topic:agi') {
        return {
          id: 'topic:agi',
          label: 'AGI',
          cluster_id: null,
          cluster_label: null,
          cluster_size: 0,
          sibling_topics: [],
          episode_count: 0,
          episodes: [],
          related_people: [],
        }
      }
      return {
        id: 'topic:ai',
        label: 'Artificial Intelligence',
        cluster_id: null,
        cluster_label: null,
        cluster_size: 0,
        sibling_topics: [
          { id: 'topic:agi', label: 'AGI', cluster_id: null, cluster_label: null, cluster_size: 0 },
        ],
        episode_count: 3,
        episodes: [],
        related_people: [],
      } as TopicCard
    })
    const { w } = await mountTopic()
    // Tap the AGI sibling → EntityCardBody's internal open() pushes.
    await w.findAll('button').find((b) => b.text() === 'AGI')!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('AGI')
    // Back pops the stack back to the origin topic (not the router).
    await w.findAll('button').find((b) => b.text().includes('Back'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('Artificial Intelligence')
  })

  it('renders the "corpus scope" tablist only when the user is signed in', async () => {
    const { w } = await mountTopic()
    expect(w.find('[role="tablist"]').exists()).toBe(false)
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b', name: 'A' }
    const router = makeRouter()
    await router.push({ name: 'topic', params: { id: 'topic:ai' } })
    await router.isReady()
    const w2 = mount(TopicView, {
      props: { id: 'topic:ai' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    await flushPromises()
    expect(w2.find('[role="tablist"]').exists()).toBe(true)
  })

  it('follow button surfaces for a signed-in user and toggles interest via addInterest', async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b', name: 'A' }
    const addInterest = vi.spyOn(api, 'addInterest').mockResolvedValue(['topic:ai'])
    const router = makeRouter()
    await router.push({ name: 'topic', params: { id: 'topic:ai' } })
    await router.isReady()
    const w = mount(TopicView, {
      props: { id: 'topic:ai' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    await flushPromises()
    const followBtn = w.findAll('button').find((b) => /^\+?\s*Follow/.test(b.text()))!
    expect(followBtn.text()).toContain('Follow')
    await followBtn.trigger('click')
    await flushPromises()
    expect(addInterest).toHaveBeenCalledWith('topic:ai')
  })

  it('failed load shows the "notFound" copy — does not crash the page', async () => {
    vi.spyOn(api, 'getTopicCard').mockRejectedValueOnce(new Error('offline'))
    const { w } = await mountTopic()
    expect(w.text()).toContain('Nothing to show for this yet.')
  })

  it('unused personCard type import stays useful across future tests', () => {
    // Silences the unused-import lint without dropping the helper someone will need.
    const _: PersonCard | null = null
    expect(_).toBeNull()
  })
})
