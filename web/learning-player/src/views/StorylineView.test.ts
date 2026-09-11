import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import StorylineView from './StorylineView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/storyline/:id', name: 'storyline', component: StorylineView, props: true },
      { path: '/topic/:id', name: 'topic', component: { template: '<div/>' }, props: true },
      { path: '/person/:id', name: 'person', component: { template: '<div/>' }, props: true },
      { path: '/episode/:slug', name: 'player', component: { template: '<div/>' }, props: true },
      { path: '/browse', name: 'browse', component: { template: '<div/>' } },
    ],
  })
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'getNotes').mockResolvedValue([])
  vi.spyOn(api, 'getHighlights').mockResolvedValue([])
})
afterEach(() => vi.restoreAllMocks())

async function mountView() {
  const router = makeRouter()
  await router.push({ name: 'storyline', params: { id: 'topic:energy' } })
  await router.isReady()
  const w = mount(StorylineView, {
    props: { id: 'topic:energy' },
    global: { plugins: [i18n, router] },
  })
  await flushPromises()
  return w
}

describe('StorylineView', () => {
  it('renders the storyline title, member topics and top episodes from the anchor card', async () => {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue({
      id: 'topic:energy',
      label: 'Energy',
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      theme_cluster_id: 'thc:energy',
      theme_cluster_label: 'Energy transition',
      theme_cluster_size: 2,
      theme_sibling_topics: [
        { id: 'topic:grid', label: 'Grid', cluster_id: null, cluster_label: null, cluster_size: 0 },
      ],
      episode_count: 1,
      episodes: [
        {
          slug: 'ep-1',
          title: 'The grid problem',
          feed_id: 'f',
          podcast_title: 'Show',
          publish_date: '2024-01-01',
          duration_seconds: 100,
          episode_image_url: null,
          feed_image_url: null,
          artwork_url: null,
          status: 'ready',
          summary_preview: null,
          summary_text: null,
          summary_bullets: [],
          topics: [],
          has_transcript: true,
          has_summary: false,
          has_gi: false,
          has_kg: false,
          has_bridge: false,
        },
      ],
      related_people: [{ id: 'person:jane', name: 'Jane', role: 'host' }],
    })
    const w = await mountView()
    expect(w.get('[data-testid="storyline-view"]').text()).toContain('Energy transition')
    expect(w.text()).toContain('Energy') // anchor topic as a member
    expect(w.text()).toContain('Grid') // sibling topic
    expect(w.text()).toContain('The grid problem') // top episode
    expect(w.text()).toContain('Jane') // person involved
  })

  it('offers Share once the storyline has loaded (#2036)', async () => {
    mockCard('thc:energy')
    const w = await mountView()
    expect(w.find('[data-testid="share-menu"]').exists()).toBe(true)
  })

  it('shows the empty message when the anchor card fails', async () => {
    vi.spyOn(api, 'getTopicCard').mockRejectedValue(new Error('nope'))
    const w = await mountView()
    expect(w.get('[data-testid="storyline-view"]').text()).toContain(
      en.home.storylineSheetEmpty,
    )
  })

  // Follow subscribes to the storyline's THEME CLUSTER (distinct from the heart, which favorites the
  // storyline). The e2e always skips — the fixture corpus has no `thc:` cluster — so this unit test
  // is the only guard on the toggle wiring.
  function mockCard(themeClusterId: string | null) {
    vi.spyOn(api, 'getTopicCard').mockResolvedValue({
      id: 'topic:energy',
      label: 'Energy',
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      theme_cluster_id: themeClusterId,
      theme_cluster_label: 'Energy transition',
      theme_cluster_size: 2,
      theme_sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
  }

  it('toggles the theme-cluster interest and reflects follow state on the button', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    mockCard('thc:energy')
    const interests = useInterestsStore()
    const toggle = vi.spyOn(interests, 'toggle').mockResolvedValue()
    const w = await mountView()
    const btn = w.get('[data-testid="storyline-follow"]')
    expect(btn.attributes('aria-pressed')).toBe('false')
    await btn.trigger('click')
    expect(toggle).toHaveBeenCalledWith('thc:energy')
  })

  it('reflects an already-followed cluster with aria-pressed=true', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    vi.spyOn(api, 'getUserInterests').mockResolvedValue(['thc:energy'])
    mockCard('thc:energy')
    const w = await mountView()
    expect(w.get('[data-testid="storyline-follow"]').attributes('aria-pressed')).toBe('true')
  })

  it('hides Follow when the topic has no theme cluster', async () => {
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    vi.spyOn(api, 'getUserInterests').mockResolvedValue([])
    mockCard(null)
    const w = await mountView()
    expect(w.find('[data-testid="storyline-follow"]').exists()).toBe(false)
  })

  it('shows momentum (badge) when the storyline is in the trending set, matched by thc: id (BT.4)', async () => {
    mockCard('thc:energy')
    vi.spyOn(api, 'getTrending').mockResolvedValue([
      {
        entity_id: 'thc:energy',
        kind: 'storyline',
        label: 'Energy transition',
        velocity: 2.2,
        volume: 30,
        heating_up: true,
        total: 30,
        series: [1, 3, 7],
      },
    ])
    const w = await mountView()
    const mom = w.get('[data-testid="trend-momentum"]')
    expect(mom.text()).toContain('2.2×')
    expect(mom.find('svg').exists()).toBe(true)
  })

  it('shows no momentum badge when the storyline is absent from the trending set', async () => {
    mockCard('thc:energy')
    vi.spyOn(api, 'getTrending').mockResolvedValue([]) // trending returns nothing for it
    const w = await mountView()
    expect(w.find('[data-testid="trend-momentum"]').exists()).toBe(false)
  })
})
