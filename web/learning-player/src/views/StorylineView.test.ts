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
    vi.spyOn(api, 'getStorylineCard').mockResolvedValue({
      id: 'thc:energy',
      label: 'Energy transition',
      // The MEMBERS now arrive already resolved and ranked, rather than being reassembled in the
      // view from the anchor plus `storyline_sibling_topics`.
      member_topics: [
        { id: 'topic:energy', label: 'Energy', cluster_id: null, cluster_label: null, cluster_size: 0 },
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
      related_people: [{ id: 'person:jane', name: 'Jane', kind: 'person' as const, role: 'host' }],
    })
    const w = await mountView()
    expect(w.get('[data-testid="storyline-view"]').text()).toContain('Energy transition')
    expect(w.text()).toContain('Energy') // anchor topic as a member
    expect(w.text()).toContain('Grid') // sibling topic
    expect(w.text()).toContain('The grid problem') // top episode
    expect(w.text()).toContain('Jane') // person involved
  })

  it('shows the people as TOP VOICES — the topic card\'s grid, not "Related people" chips', async () => {
    // Operator 2026-09-30: the topic card drew these people as an avatar grid, the storyline page
    // as plain chips — same `related_people`, two treatments. Both now render TopVoices.
    const people = Array.from({ length: 10 }, (_, i) => ({
      id: `person:p${i}`,
      name: `Person ${i}`,
      kind: 'person' as const,
      image_url: i === 0 ? 'https://img.example/p0.jpg' : null,
    }))
    vi.spyOn(api, 'getStorylineCard').mockResolvedValue({
      id: 'thc:energy',
      label: 'Energy transition',
      // A member is required: the view renders its body only once the grouping HAS topics —
      // an empty grouping is the "couldn't load" state, so top voices would never show.
      member_topics: [
        { id: 'topic:energy', label: 'Energy', cluster_id: null, cluster_label: null, cluster_size: 0 },
      ],
      episode_count: 0,
      episodes: [],
      related_people: people,
    })
    const w = await mountView()
    const grid = w.get('[data-testid="ec-top-voices"]')
    expect(grid.text()).toContain(en.ec.topVoices)
    expect(w.text()).not.toContain(en.ec.relatedPeople)
    const voices = grid.findAll('[data-testid="ec-top-voice"]')
    expect(voices).toHaveLength(8) // ranked; the top 8, as on the topic card
    // A page keeps real links: an address a long-press can open.
    expect(voices[0].attributes('href')).toBe('/person/person:p0')
    expect(voices[0].find('img').attributes('src')).toBe('https://img.example/p0.jpg')
  })

  it('offers Share once the storyline has loaded (#2036)', async () => {
    mockCard('thc:energy')
    const w = await mountView()
    expect(w.find('[data-testid="share-menu"]').exists()).toBe(true)
  })

  it('shows the empty message when the anchor card fails', async () => {
    vi.spyOn(api, 'getStorylineCard').mockRejectedValue(new Error('nope'))
    const w = await mountView()
    expect(w.get('[data-testid="storyline-view"]').text()).toContain(
      en.home.storylineSheetEmpty,
    )
  })

  // Follow subscribes to the storyline's THEME CLUSTER (distinct from the heart, which favorites the
  // storyline). The e2e covers it too since the fixture corpus gained `thc:managing-risk`
  // (storyline.spec.ts, 2026-09-25) — this comment used to say the e2e always skips, which stopped
  // being true then; this unit test guards the toggle wiring without a server.
  function mockCard(storylineId: string | null) {
    // `null` = this id resolves to no storyline, so the endpoint 404s and the view shows its empty
    // state with no Follow — the case the old mock expressed as a null `storyline_id`.
    if (storylineId === null) {
      vi.spyOn(api, 'getStorylineCard').mockRejectedValue(new Error('404'))
      return
    }
    vi.spyOn(api, 'getStorylineCard').mockResolvedValue({
      // The card's id IS the follow token now — the view no longer learns it from a topic card.
      id: storylineId,
      label: 'Energy transition',
      member_topics: [
        { id: 'topic:energy', label: 'Energy', cluster_id: null, cluster_label: null, cluster_size: 0 },
      ],
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
