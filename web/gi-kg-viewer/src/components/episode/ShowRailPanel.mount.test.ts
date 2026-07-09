// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ShowRailPanel from './ShowRailPanel.vue'
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
    has_gi: true,
    has_kg: true,
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
}

function stubApi(
  nextCursor: string | null = null,
  signals: unknown = SIGNALS,
): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/corpus/feed-signals')) return res(signals)
      if (url.includes('/api/corpus/feeds')) return res({ path: '/corpus', feeds: [FEED] })
      if (url.includes('/api/corpus/episodes'))
        return res({ path: '/corpus', feed_id: 'alpha', items: EPISODES, next_cursor: nextCursor })
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
    expect(w.find('[data-testid="show-rail-episode-0"]').text()).toContain('GI')
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

  it('hides the signals block when the feed has no topics or people', async () => {
    stubApi(null, { path: '/corpus', feed_id: 'alpha', episode_count: 0, top_topics: [], key_people: [] })
    const w = mount(ShowRailPanel)
    await flushPromises()
    expect(w.find('[data-testid="show-rail-signals"]').exists()).toBe(false)
  })
})
