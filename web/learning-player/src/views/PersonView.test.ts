import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { PersonCard } from '../services/types'
import { useAuthStore } from '../stores/auth'
import PersonView from './PersonView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/person/:id', name: 'person', component: PersonView, props: true },
      // Cross-links from EntityCardBody push their own back-stack, not routes,
      // BUT searchLibrary / episode links / cross-open topic pushes do resolve
      // through the router — register stubs so tests don't blow up.
      { path: '/search', name: 'search', component: { template: '<div/>' } },
      { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
      { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
      { path: '/topic/:id', name: 'topic', component: { template: '<div/>' }, props: true },
    ],
  })
}

beforeEach(() => {
  vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({})
  vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
  vi.spyOn(api, 'getPersonCard').mockResolvedValue({
    id: 'person:jane-doe',
    label: 'Jane Doe',
    role: 'host',
    episode_count: 5,
    episodes: [],
    related_people: [
      { id: 'person:bob-smith', name: 'Bob Smith', kind: 'person' },
    ],
    related_topics: [
      { id: 'topic:ai', label: 'AI', cluster_id: null, cluster_label: null, cluster_size: 0 },
    ],
  })
})
afterEach(() => vi.restoreAllMocks())

async function mountPerson(id = 'person:jane-doe') {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: 'person', params: { id } })
  await router.isReady()
  const w = mount(PersonView, {
    props: { id },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await flushPromises()
  return { w, router }
}

describe('PersonView (#1261-6)', () => {
  it('fetches the person card via the route param and renders the person label', async () => {
    const { w } = await mountPerson()
    expect(api.getPersonCard).toHaveBeenCalledWith('person:jane-doe', undefined)
    expect(w.find('[data-testid="person-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Jane Doe')
  })

  it('renders in inline mode — Back button, no Open-in-page link', async () => {
    const { w } = await mountPerson()
    expect(w.text()).toContain('Back')
    expect(w.find('[data-testid="ec-open-in-page"]').exists()).toBe(false)
  })

  it('renders the person role badge for a signed-in view', async () => {
    const { w } = await mountPerson()
    // Role badge always renders (auth-independent — mirrors operator viewer).
    const roleBadge = w.find('[data-testid="ec-person-role"]')
    expect(roleBadge.exists()).toBe(true)
    expect(roleBadge.attributes('data-role')).toBe('host')
    expect(roleBadge.text().toLowerCase()).toContain('host')
  })

  it('renders related-topic chips linking (via internal open() stack push) to a topic card', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue({
      id: 'topic:ai',
      label: 'AI',
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
    const { w } = await mountPerson()
    const topicChip = w.findAll('button').find((b) => b.text() === 'AI')
    expect(topicChip).toBeTruthy()
    await topicChip!.trigger('click')
    await flushPromises()
    // Internal stack push — the header title flips to the topic.
    expect(w.text()).toContain('AI')
    // Back returns to the person origin.
    await w.findAll('button').find((b) => b.text().includes('Back'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('Jane Doe')
  })

  it('follow button surfaces for a signed-in user and toggles interest via addInterest', async () => {
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: 'u1', email: 'a@b', name: 'A' }
    const addInterest = vi.spyOn(api, 'addInterest').mockResolvedValue(['person:jane-doe'])
    const router = makeRouter()
    await router.push({ name: 'person', params: { id: 'person:jane-doe' } })
    await router.isReady()
    const w = mount(PersonView, {
      props: { id: 'person:jane-doe' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
    await flushPromises()
    const followBtn = w.findAll('button').find((b) => /^\+?\s*Follow/.test(b.text()))!
    expect(followBtn.text()).toContain('Follow')
    await followBtn.trigger('click')
    await flushPromises()
    expect(addInterest).toHaveBeenCalledWith('person:jane-doe')
  })

  it('failed load shows the notFound copy — does not crash', async () => {
    vi.spyOn(api, 'getPersonCard').mockRejectedValueOnce(new Error('offline'))
    const { w } = await mountPerson()
    expect(w.text()).toContain('Nothing to show for this yet.')
  })

  it('keeps the PersonCard type import in scope for cross-file symmetry', () => {
    const _: PersonCard | null = null
    expect(_).toBeNull()
  })
})
