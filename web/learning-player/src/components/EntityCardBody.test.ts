import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary, PersonCard, TopicCard } from '../services/types'
import { useAuthStore } from '../stores/auth'
import EntityCardBody from './EntityCardBody.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/topic/:id', name: 'topic', component: { template: '<div/>' }, props: true },
    { path: '/person/:id', name: 'person', component: { template: '<div/>' }, props: true },
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

function ep(slug: string, feed_id: string, title = slug): EpisodeSummary {
  return {
    slug,
    title,
    feed_id,
    podcast_title: feed_id,
    publish_date: '2024-03-10',
    duration_seconds: null,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    status: 'ready',
    summary_preview: null,
    summary_text: null,
    summary_bullets: [],
    topics: [],
    has_transcript: true,
    has_summary: true,
    has_gi: false,
    has_kg: true,
    has_bridge: false,
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
  // The embedded EntitySignals fetches corpus enrichment; keep these tests off
  // the network (its own coverage lives in EntitySignals.test.ts).
  vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({})
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

  it('renders the theme-cluster identity + theme members (co-occurrence)', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(
      topicCard({
        cluster_id: null,
        cluster_label: null,
        cluster_size: 0,
        theme_cluster_id: 'thc:sanctions',
        theme_cluster_label: 'sanctions',
        theme_cluster_size: 3,
        theme_sibling_topics: [
          {
            id: 'topic:oil',
            label: 'oil',
            cluster_id: null,
            cluster_label: null,
            cluster_size: 0,
          },
        ],
      }),
    )
    const w = mountAuthed({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()
    // "Theme ·" identity line — distinct from the semantic "Similar ·".
    expect(w.text()).toContain('Theme · sanctions')
    const themeMembers = w.find('[data-testid="ec-theme-members"]')
    expect(themeMembers.exists()).toBe(true)
    expect(themeMembers.text()).toContain('oil')
  })

  it('follows the whole storyline (thc:) via the Theme-line toggle, distinct from the topic follow', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    vi.spyOn(api, 'addInterest').mockResolvedValue(['thc:sanctions'])
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(
      topicCard({
        theme_cluster_id: 'thc:sanctions',
        theme_cluster_label: 'sanctions',
        theme_cluster_size: 3,
      }),
    )
    const w = mountAuthed({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()

    const btn = () => w.find('[data-testid="ec-follow-storyline"]')
    expect(btn().exists()).toBe(true)
    expect(btn().attributes('aria-pressed')).toBe('false')

    await btn().trigger('click')
    await flushPromises()

    // Follows the theme-cluster token, NOT the topic id (that's the header button's job).
    expect(api.addInterest).toHaveBeenCalledWith('thc:sanctions')
    expect(btn().attributes('aria-pressed')).toBe('true')
    expect(btn().text()).toContain('Following storyline')
  })

  it('starts Following storyline when the thc: token is already an interest', async () => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['thc:sanctions'])
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(
      topicCard({
        theme_cluster_id: 'thc:sanctions',
        theme_cluster_label: 'sanctions',
        theme_cluster_size: 3,
      }),
    )
    const w = mountAuthed({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()
    const btn = w.find('[data-testid="ec-follow-storyline"]')
    expect(btn.attributes('aria-pressed')).toBe('true')
    expect(btn.text()).toContain('Following storyline')
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

describe('EntityCardBody — your-corpus lens (P3 #1125)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
  })

  it('My corpus refetches the card scoped to the heard set', async () => {
    const getPerson = vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    // default load is unscoped
    expect(getPerson).toHaveBeenLastCalledWith('person:jane-doe', undefined)
    // tap "My corpus" → refetch with scope=mine
    await w.findAll('[role="tab"]').find((b) => b.text() === 'My corpus')!.trigger('click')
    await flushPromises()
    expect(getPerson).toHaveBeenLastCalledWith('person:jane-doe', 'mine')
  })

  it('hides the scope toggle when signed out', async () => {
    setActivePinia(createPinia())
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mount(EntityCardBody, {
      props: { kind: 'person', id: 'person:jane-doe', variant: 'overlay' },
      global: { plugins: [i18n, router] },
    })
    await flushPromises()
    expect(w.find('[role="tablist"]').exists()).toBe(false)
  })
})

describe('EntityCardBody — speaker role badge (#3)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
  })

  it('shows a Host badge with the ringed-emphasis class for a host', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard({ role: 'host' }))
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    const badge = w.find('[data-testid="ec-person-role"]')
    expect(badge.exists()).toBe(true)
    expect(badge.text()).toBe('Host')
    expect(badge.attributes('data-role')).toBe('host')
    expect(badge.classes()).toContain('ring-person')
  })

  it('shows a Guest badge without the host ring', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard({ role: 'guest' }))
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    const badge = w.find('[data-testid="ec-person-role"]')
    expect(badge.text()).toBe('Guest')
    expect(badge.attributes('data-role')).toBe('guest')
    expect(badge.classes()).not.toContain('ring-person')
  })

  it('renders no badge when the person has no role', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard({ role: null }))
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    expect(w.find('[data-testid="ec-person-role"]').exists()).toBe(false)
  })

  it('renders no role badge for a topic card', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    const w = mountAuthed({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()
    expect(w.find('[data-testid="ec-person-role"]').exists()).toBe(false)
  })
})

describe('EntityCardBody — per-show roles (host of one, guest of another)', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
  })

  it('shows a "Host of" section and drops the hosted show\'s episodes from the list', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(
      personCard({
        role: 'host',
        shows: [
          { feed_id: 'showA', title: 'Show A', role: 'host', episode_count: 2 },
          { feed_id: 'showB', title: 'Show B', role: 'guest', episode_count: 1 },
        ],
        episode_count: 3,
        episodes: [ep('a1', 'showA'), ep('a2', 'showA'), ep('b1', 'showB', 'Guest spot')],
      }),
    )
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()

    // "Host of" lists the hosted show and links to its show page.
    const hostShows = w.find('[data-testid="ec-host-shows"]')
    expect(hostShows.exists()).toBe(true)
    expect(hostShows.text()).toContain('Show A')
    expect(hostShows.find('a').attributes('href')).toContain('/podcast/showA')

    // The episode list drops Show A's back-catalogue and shows only the other-show appearance.
    expect(w.text()).toContain('Also appears in')
    expect(w.text()).toContain('Guest spot')
    expect(w.text()).not.toContain('a1')
    expect(w.text()).not.toContain('a2')
  })

  it('renders no "Host of" section when the person hosts nothing', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(
      personCard({
        role: 'guest',
        shows: [{ feed_id: 'showB', title: 'Show B', role: 'guest', episode_count: 1 }],
        episode_count: 1,
        episodes: [ep('b1', 'showB')],
      }),
    )
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    expect(w.find('[data-testid="ec-host-shows"]').exists()).toBe(false)
    // Non-host keeps the plain "In N episodes" heading (all episodes shown).
    expect(w.text()).toContain('In 1 episode')
  })
})

// #1261-9: "Open in page" link (overlay-mode escape hatch to standalone page)
describe('EntityCardBody — Open in page link', () => {
  it('overlay mode: renders a link pointing at /topic/:id for topic entities', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    const w = mountAuthed({ kind: 'topic', id: 'topic:ai' })
    await flushPromises()
    const link = w.get('[data-testid="ec-open-in-page"]')
    expect(link.attributes('href')).toBe('/topic/topic:ai')
    expect(link.text()).toContain('Open in page')
  })

  it('overlay mode: renders a link pointing at /person/:id for person entities', async () => {
    vi.spyOn(api, 'getPersonCard').mockResolvedValue(personCard())
    const w = mountAuthed({ kind: 'person', id: 'person:jane-doe' })
    await flushPromises()
    const link = w.get('[data-testid="ec-open-in-page"]')
    expect(link.attributes('href')).toBe('/person/person:jane-doe')
  })

  it('overlay mode: clicking the link emits close so the modal dismisses as we navigate', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
    const w = mount(EntityCardBody, {
      props: { kind: 'topic', id: 'topic:ai', variant: 'overlay' },
      global: { plugins: [i18n, router] },
    })
    await flushPromises()
    await w.get('[data-testid="ec-open-in-page"]').trigger('click')
    // 'close' is emitted so the parent EntityCard can teardown before nav.
    expect(w.emitted('close')).toBeTruthy()
  })

  it('inline mode: does NOT render the link (already on the page / inside a panel)', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue(topicCard())
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
    const w = mount(EntityCardBody, {
      props: { kind: 'topic', id: 'topic:ai', variant: 'inline' },
      global: { plugins: [i18n, router] },
    })
    await flushPromises()
    expect(w.find('[data-testid="ec-open-in-page"]').exists()).toBe(false)
  })
})
