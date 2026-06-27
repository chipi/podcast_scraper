import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { PersonCard, TopicCard } from '../services/types'
import { useAuthStore } from '../stores/auth'
import EntityCardBody from './EntityCardBody.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
  ],
})

function personCard(over: Partial<PersonCard> = {}): PersonCard {
  return {
    id: 'person:jane-doe',
    label: 'Jane Doe',
    episode_count: 1,
    episodes: [],
    related_people: [],
    related_topics: [],
    ...over,
  }
}

function topicCard(over: Partial<TopicCard> = {}): TopicCard {
  return {
    id: 'topic:ai',
    label: 'AI',
    cluster_id: 'tc:ai',
    cluster_label: 'AI',
    cluster_size: 2,
    sibling_topics: [],
    episode_count: 1,
    episodes: [],
    related_people: [],
    ...over,
  }
}

/** Mount with a signed-in user so the Follow control renders (it gates on auth.isAuthenticated). */
function mountAuthed(props: { kind: 'person' | 'topic'; id: string }) {
  setActivePinia(createPinia())
  const auth = useAuthStore()
  auth.user = { user_id: 'u_1', email: 'd@l', name: 'Dev' } // → isAuthenticated true
  return mount(EntityCardBody, {
    props: { ...props, variant: 'overlay' },
    global: { plugins: [i18n, router] },
  })
}

const followBtn = (w: ReturnType<typeof mountAuthed>) =>
  w.findAll('button').find((b) => /Follow|Following/.test(b.text()))!

beforeEach(() => {
  vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

describe('EntityCardBody — Follow control', () => {
  it('renders the Follow button for an authed user (person)', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    const btn = followBtn(w)
    expect(btn.text()).toContain('Follow')
    expect(btn.attributes('aria-pressed')).toBe('false')
  })

  it('clicking Follow calls interests.toggle with the current entity id and flips to Following', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    vi.spyOn(api, 'addInterest').mockResolvedValue(['person:jane-doe'])
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()

    await followBtn(w).trigger('click')
    await flushPromises()

    // The store mutation routes through the real api write for this token.
    expect(api.addInterest).toHaveBeenCalledWith('person:jane-doe')
    const btn = followBtn(w)
    expect(btn.text()).toContain('Following')
    expect(btn.attributes('aria-pressed')).toBe('true')
  })

  it('starts as Following when the token is already an interest, and unfollow flips back', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['topic:ai'])
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    vi.spyOn(api, 'removeInterest').mockResolvedValue([])
    const w = mountAuthed({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()

    expect(followBtn(w).text()).toContain('Following')
    expect(followBtn(w).attributes('aria-pressed')).toBe('true')

    await followBtn(w).trigger('click')
    await flushPromises()

    expect(api.removeInterest).toHaveBeenCalledWith('topic:ai')
    expect(followBtn(w).text()).toContain('Follow')
    expect(followBtn(w).attributes('aria-pressed')).toBe('false')
  })

  it('hides the Follow control when signed out', async () => {
    setActivePinia(createPinia()) // fresh pinia, no user → signed out
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mount(EntityCardBody, {
      props: { kind: 'person', id: 'person:jane-doe', variant: 'overlay' },
      global: { plugins: [i18n, router] },
    })
    await flushPromises()
    expect(w.findAll('button').some((b) => /Follow|Following/.test(b.text()))).toBe(false)
  })
})
