// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ShowRailPanel from './ShowRailPanel.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

/**
 * UXS-015 / RFC-104 — the Show rail panel. Reads subject.feedId (set by focusShow),
 * re-fetches the feed (header) + episodes, and opens an episode via focusEpisode
 * (which pushes the show onto the Back stack).
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

const FEED = {
  feed_id: 'alpha',
  display_title: 'Alpha Show',
  episode_count: 2,
  description: 'A great show.',
  rss_url: 'http://x/rss.xml',
  image_url: 'http://x/art.jpg',
}
const EPISODES = [
  {
    metadata_relative_path: 'metadata/a1.metadata.json',
    feed_id: 'alpha',
    episode_id: 'episode:a1',
    episode_title: 'Alpha Episode One',
    publish_date: '2026-06-01',
    summary_preview: 'The full recap for episode one goes here.',
    has_gi: true,
    has_kg: true,
    kg_relative_path: 'metadata/a1.kg.json',
    cil_digest_topics: [
      { topic_id: 'topic:ai', label: 'AI', in_topic_cluster: true },
      { topic_id: 'topic:policy', label: 'policy', in_topic_cluster: false },
    ],
  },
  {
    metadata_relative_path: 'metadata/a2.metadata.json',
    feed_id: 'alpha',
    episode_id: 'episode:a2',
    episode_title: 'Alpha Episode Two',
    publish_date: '2026-05-01',
  },
]

const SIGNALS = {
  path: '/corpus',
  feed_id: 'alpha',
  episode_count: 2,
  top_topics: [
    { topic_id: 'topic:ai', label: 'AI', episode_count: 2 },
    { topic_id: 'topic:ethics', label: 'Ethics', episode_count: 1 },
  ],
  key_people: [{ person_id: 'person:jane', name: 'Jane Doe', episode_count: 2 }],
  recurring_guests: [{ person_id: 'person:jane', name: 'Jane Doe', episode_count: 2 }],
  dominant_themes: [{ theme_id: 'thc:ai-stuff', label: 'AI stuff', topic_count: 3 }],
  trending_topics: [{ topic_id: 'topic:ai', label: 'AI', velocity: 2.5, episode_count: 2 }],
  grounding: { grounded_insights: 8, total_insights: 10, rate: 0.8, people_count: 3 },
}

let lastEpisodesUrl = ''

function stubApi(
  nextCursor: string | null = null,
  signals: unknown = SIGNALS,
): void {
  lastEpisodesUrl = ''
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/corpus/feed-signals')) return res(signals)
      if (url.includes('/api/corpus/feeds')) return res({ path: '/corpus', feeds: [FEED] })
      if (url.includes('/api/corpus/episodes')) {
        lastEpisodesUrl = url
        return res({ path: '/corpus', feed_id: 'alpha', items: EPISODES, next_cursor: nextCursor })
      }
      return res({}, 404)
    }),
  )
}

beforeEach(() => {
  setActivePinia(createPinia())
  useShellStore().corpusPath = '/corpus'
  useSubjectStore().focusShow('alpha', { uiTitle: 'Alpha Show' })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ShowRailPanel — header + episodes', () => {
  it('renders the show header (title, count, RSS) from the feed', async () => {
    stubApi()
    const w = mount(ShowRailPanel)
    await flushPromises()

    const panel = w.find('[data-testid="show-rail-panel"]')
    expect(panel.text()).toContain('Alpha Show')
    expect(panel.text()).toContain('2 episodes')
    expect(w.find('a[href="http://x/rss.xml"]').exists()).toBe(true)
  })

  it('lists the show episodes with GI/KG badges', async () => {
    stubApi()
    const w = mount(ShowRailPanel)
    await flushPromises()

    expect(w.find('[data-testid="show-rail-episode-0"]').text()).toContain('Alpha Episode One')
    expect(w.find('[data-testid="show-rail-episode-1"]').text()).toContain('Alpha Episode Two')
    // GI/KG badges are siblings of the (clickable) title area, not nested inside it.
    expect(w.get('[data-testid="show-rail-panel"]').text()).toContain('GI')
  })

  it('opens an episode via subject.focusEpisode (Back-to-show set up)', async () => {
    stubApi()
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusEpisode')
    const w = mount(ShowRailPanel)
    await flushPromises()

    await w.find('[data-testid="show-rail-episode-0"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('metadata/a1.metadata.json', {
      uiTitle: 'Alpha Episode One',
      episodeId: 'episode:a1',
    })
    // focusEpisode from a show pushes Back history.
    expect(subject.kind).toBe('episode')
    expect(subject.canGoBack).toBe(true)
  })

  it('shows Load more only when a cursor is present', async () => {
    stubApi('cursor-2')
    const w = mount(ShowRailPanel)
    await flushPromises()
    expect(w.find('[data-testid="show-rail-load-more"]').exists()).toBe(true)
  })
})

describe('ShowRailPanel — show signals', () => {
  it('renders top topics + key people with episode counts', async () => {
    stubApi()
    const w = mount(ShowRailPanel)
    await flushPromises()

    const sig = w.find('[data-testid="show-rail-signals"]')
    expect(sig.exists()).toBe(true)
    const topics = w.findAll('[data-testid="show-rail-topic"]')
    expect(topics.map((t) => t.text().replace(/\s+/g, ' '))).toEqual(['AI · 2', 'Ethics · 1'])
    const people = w.findAll('[data-testid="show-rail-person"]')
    expect(people.map((p) => p.text().replace(/\s+/g, ' '))).toEqual(['Jane Doe · 2'])
  })

  it('opens a topic chip via subject.focusTopic (Back-to-show set up)', async () => {
    stubApi()
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusTopic')
    const w = mount(ShowRailPanel)
    await flushPromises()

    await w.findAll('[data-testid="show-rail-topic"]')[0].trigger('click')
    expect(spy).toHaveBeenCalledWith('topic:ai')
    expect(subject.kind).toBe('graph-node')
    expect(subject.canGoBack).toBe(true)
  })

  it('opens a person chip via subject.focusPerson', async () => {
    stubApi()
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusPerson')
    const w = mount(ShowRailPanel)
    await flushPromises()

    await w.find('[data-testid="show-rail-person"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('person:jane')
  })

  it('renders the enrichment aggregates: themes, trending, grounding, recurring guests', async () => {
    stubApi()
    const w = mount(ShowRailPanel)
    await flushPromises()

    expect(w.get('[data-testid="show-rail-grounding"]').text()).toContain('80%')
    expect(w.get('[data-testid="show-rail-theme"]').text().replace(/\s+/g, ' ')).toBe('AI stuff · 3')
    expect(w.get('[data-testid="show-rail-trending"]').text()).toContain('2.5×')
    expect(w.get('[data-testid="show-rail-recurring"]').text().replace(/\s+/g, ' ')).toBe(
      'Jane Doe · 2',
    )
  })

  it('opens a trending chip via subject.focusTopic', async () => {
    stubApi()
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusTopic')
    const w = mount(ShowRailPanel)
    await flushPromises()
    await w.get('[data-testid="show-rail-trending"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('topic:ai')
  })

  it('opens a theme chip (the thc: cluster node) via subject.focusTopic', async () => {
    stubApi()
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusTopic')
    const w = mount(ShowRailPanel)
    await flushPromises()
    await w.get('[data-testid="show-rail-theme"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('thc:ai-stuff')
  })

  it('hides the signals block when the feed has no topics or people', async () => {
    stubApi(null, { path: '/corpus', feed_id: 'alpha', episode_count: 0, top_topics: [], key_people: [] })
    const w = mount(ShowRailPanel)
    await flushPromises()
    expect(w.find('[data-testid="show-rail-signals"]').exists()).toBe(false)
  })
})

describe('ShowRailPanel — episode rows, sort, graph', () => {
  it('shows each episode full summary + digest-parity topic pills (cluster-coloured)', async () => {
    stubApi()
    const w = mount(ShowRailPanel)
    await flushPromises()

    expect(w.get('[data-testid="show-rail-episode-0"]').text()).toContain(
      'The full recap for episode one',
    )
    const pills = w.get('[data-testid="show-rail-episode-pills-0"]').findAll('button')
    expect(pills.map((p) => p.text())).toEqual(['AI', 'policy'])
    // The clustered pill (AI) gets the kg cluster chrome.
    expect(pills[0].classes().join(' ')).toContain('border-kg')
    // Opt-in pills: the episodes request carried with_cil_topics.
    expect(lastEpisodesUrl).toContain('with_cil_topics=true')
  })

  it('a topic pill opens that topic via subject.focusTopic', async () => {
    stubApi()
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusTopic')
    const w = mount(ShowRailPanel)
    await flushPromises()

    await w.get('[data-testid="show-rail-episode-pills-0"]').findAll('button')[1].trigger('click')
    expect(spy).toHaveBeenCalledWith('topic:policy')
  })

  it('the sort control refetches episodes oldest-first', async () => {
    stubApi()
    const w = mount(ShowRailPanel)
    await flushPromises()
    expect(lastEpisodesUrl).toContain('sort=newest')

    await w.get('[data-testid="show-rail-sort-oldest"]').trigger('click')
    await flushPromises()
    expect(lastEpisodesUrl).toContain('sort=oldest')
  })

  it('the graph icon switches to Graph and loads the show KGs', async () => {
    stubApi()
    const artifacts = useArtifactsStore()
    const spy = vi.spyOn(artifacts, 'appendRelativeArtifacts').mockResolvedValue(undefined)
    const w = mount(ShowRailPanel)
    await flushPromises()

    await w.get('[data-testid="show-rail-open-graph"]').trigger('click')
    expect(w.emitted('switch-main-tab')![0]).toEqual(['graph'])
    expect(spy).toHaveBeenCalledWith(['metadata/a1.kg.json'])
  })
})
