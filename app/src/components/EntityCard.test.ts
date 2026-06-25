import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { PersonCard, TopicCard } from '../services/types'
import EntityCard from './EntityCard.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
  ],
})

const mountCard = (props: { kind: 'person' | 'topic'; id: string }) =>
  mount(EntityCard, { props, global: { plugins: [i18n, router] } })

function personCard(over: Partial<PersonCard> = {}): PersonCard {
  return {
    id: 'person:jane-doe',
    label: 'Jane Doe',
    episode_count: 2,
    episodes: [
      {
        slug: 'ep-1', title: 'First Episode', feed_id: 'f', podcast_title: 'Show',
        publish_date: '2024-06-01', duration_seconds: null, episode_image_url: null,
        feed_image_url: null, artwork_url: null, status: 'ready', summary_preview: null,
        summary_bullets: [], topics: [], has_transcript: true, has_summary: false,
        has_gi: false, has_kg: true, has_bridge: false,
      },
    ],
    related_people: [{ id: 'person:bob', name: 'Bob', kind: 'person' }],
    related_topics: [
      { id: 'topic:ai', label: 'AI', cluster_id: 'tc:ai', cluster_label: 'AI', cluster_size: 2 },
    ],
    ...over,
  }
}

function topicCard(over: Partial<TopicCard> = {}): TopicCard {
  return {
    id: 'topic:ai',
    label: 'AI',
    cluster_id: 'tc:ai',
    cluster_label: 'Artificial Intelligence',
    cluster_size: 2,
    sibling_topics: [
      { id: 'topic:ml', label: 'Machine Learning', cluster_id: 'tc:ai', cluster_label: 'AI', cluster_size: 2 },
    ],
    episode_count: 3,
    episodes: [
      {
        slug: 'ep-ai', title: 'AI Episode', feed_id: 'f', podcast_title: 'Show',
        publish_date: '2024-06-01', duration_seconds: null, episode_image_url: null,
        feed_image_url: null, artwork_url: null, status: 'ready', summary_preview: null,
        summary_bullets: [], topics: [], has_transcript: true, has_summary: false,
        has_gi: false, has_kg: true, has_bridge: false,
      },
    ],
    related_people: [{ id: 'person:jane-doe', name: 'Jane Doe', kind: 'person' }],
    ...over,
  }
}

afterEach(() => vi.restoreAllMocks())

describe('EntityCard', () => {
  it('renders a person card: label, episode count, episodes, related chips', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mountCard({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    expect(w.text()).toContain('Jane Doe')
    expect(w.text()).toContain('In 2 episodes')
    expect(w.text()).toContain('First Episode')
    expect(w.text()).toContain('Bob') // related person chip
    expect(w.text()).toContain('AI') // related topic chip
    expect(w.findAll('a').map((a) => a.attributes('href'))).toContain('/episode/ep-1')
  })

  it('renders a topic card: theme line, sibling themes, episode-about count', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    const w = mountCard({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()
    expect(w.text()).toContain('Artificial Intelligence') // cluster theme line
    expect(w.text()).toContain('More in this theme')
    expect(w.text()).toContain('Machine Learning') // sibling chip
    expect(w.text()).toContain('Discussed in 3 episodes')
  })

  it('the library search lives inside the card (button → search route, then closes)', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const push = vi.spyOn(router, 'push')
    const w = mountCard({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('Search the library'))!.trigger('click')
    expect(push).toHaveBeenCalledWith({ name: 'search', query: { q: 'Jane Doe' } })
    expect(w.emitted('close')).toBeTruthy()
  })

  it('is re-entrant: tapping a related chip walks to that entity and back', async () => {
    const getPerson = vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const getTopic = vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    const w = mountCard({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    // Tap the related AI topic chip → loads the topic card in place.
    await w.findAll('button').find((b) => b.text() === 'AI')!.trigger('click')
    await flushPromises()
    expect(getTopic).toHaveBeenCalledWith('topic:ai')
    expect(w.text()).toContain('More in this theme') // topic view now shown
    // Back → returns to the person.
    await w.find('button[aria-label="Back"]').trigger('click')
    await flushPromises()
    expect(getPerson).toHaveBeenCalledTimes(2) // reloaded on return
    expect(w.text()).toContain('In 2 episodes')
  })

  it('emits close on the dimmed backdrop and the ✕ button', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mountCard({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    await w.find('[role="dialog"]').trigger('click') // self-click on backdrop
    expect(w.emitted('close')).toBeTruthy()
  })

  it('moves focus into the dialog on open and restores it on close (modal a11y)', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const anchor = document.createElement('button')
    document.body.appendChild(anchor)
    anchor.focus()
    expect(document.activeElement).toBe(anchor)
    const w = mount(EntityCard, {
      props: { kind: 'person', id: 'person:jane-doe' },
      global: { plugins: [i18n, router] },
      attachTo: document.body,
    })
    await flushPromises()
    const dialog = w.find('[role="dialog"]').element
    expect(dialog.contains(document.activeElement)).toBe(true)
    w.unmount()
    expect(document.activeElement).toBe(anchor)
    anchor.remove()
  })

  it('shows a graceful message when the entity has no footprint (404)', async () => {
    vi.spyOn(api, 'getPersonCard').mockRejectedValue(new api.ApiError(404, 'nope'))
    const w = mountCard({ kind: 'person', id: 'person:ghost' })
    await flushPromises()
    expect(w.text()).toContain('Nothing to show')
  })
})
