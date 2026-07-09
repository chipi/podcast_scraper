// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ShowDetailView from './ShowDetailView.vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'

/**
 * UXS-015 / RFC-104 — mount tests for the show-detail view. Verifies the header
 * derives from the feed prop, episodes load from /api/corpus/episodes, an episode
 * click cross-links via subject.focusEpisode (the shared Library path), Back emits,
 * and the empty state renders when a show has no episodes.
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubEpisodes(items: unknown[], nextCursor: string | null = null): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/corpus/episodes'))
        return res({ path: '/corpus', feed_id: 'big', items, next_cursor: nextCursor })
      return res({}, 404)
    }),
  )
}

const FEED: CorpusFeedItem = {
  feed_id: 'big',
  display_title: 'Big Show',
  episode_count: 2,
  description: 'A great show.',
  rss_url: 'http://x/rss.xml',
}

const EPISODES = [
  {
    metadata_relative_path: 'feeds/big/run_1/metadata/ep1.metadata.json',
    feed_id: 'big',
    episode_id: 'episode:ep1',
    episode_title: 'Episode One',
    publish_date: '2026-06-15T00:00:00Z',
    summary_title: 'The first one',
    has_gi: true,
    has_kg: true,
  },
  {
    metadata_relative_path: 'feeds/big/run_1/metadata/ep2.metadata.json',
    feed_id: 'big',
    episode_id: 'episode:ep2',
    episode_title: 'Episode Two',
    publish_date: '2026-05-15T00:00:00Z',
  },
]

beforeEach(() => {
  setActivePinia(createPinia())
  useShellStore().corpusPath = '/corpus'
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ShowDetailView — mount + behaviour', () => {
  it('renders the header from the feed prop', async () => {
    stubEpisodes(EPISODES)
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    const header = w.find('[data-testid="show-detail"]')
    expect(header.text()).toContain('Big Show')
    expect(header.text()).toContain('2 episodes')
    expect(w.find('a[href="http://x/rss.xml"]').exists()).toBe(true)
  })

  it('renders the show episodes with GI/KG badges', async () => {
    stubEpisodes(EPISODES)
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    expect(w.find('[data-testid="show-detail-episode-0"]').text()).toContain('Episode One')
    expect(w.find('[data-testid="show-detail-episode-1"]').text()).toContain('Episode Two')
    // ep1 carries GI + KG badges; ep2 does not.
    expect(w.find('[data-testid="show-detail-episode-0"]').text()).toContain('GI')
    expect(w.find('[data-testid="show-detail-episode-0"]').text()).toContain('KG')
    expect(w.find('[data-testid="show-detail-episode-1"]').text()).not.toContain('GI')
  })

  it('cross-links an episode via subject.focusEpisode', async () => {
    stubEpisodes(EPISODES)
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusEpisode')
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    await w.find('[data-testid="show-detail-episode-0"]').trigger('click')
    expect(spy).toHaveBeenCalledWith('feeds/big/run_1/metadata/ep1.metadata.json', {
      uiTitle: 'Episode One',
      episodeId: 'episode:ep1',
    })
  })

  it('emits back when the Back control is clicked', async () => {
    stubEpisodes(EPISODES)
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    await w.find('[data-testid="show-detail-back"]').trigger('click')
    expect(w.emitted('back')).toBeTruthy()
  })

  it('shows the empty state when a show has no episodes', async () => {
    stubEpisodes([])
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    expect(w.find('[data-testid="show-detail-empty"]').exists()).toBe(true)
  })

  it('reveals Load more only when a cursor is present', async () => {
    stubEpisodes(EPISODES, 'cursor-2')
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    expect(w.find('[data-testid="show-detail-load-more"]').exists()).toBe(true)
  })

  it('appends the next page and hides Load more when the cursor is exhausted', async () => {
    let call = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        call += 1
        return call === 1
          ? res({ path: '/corpus', feed_id: 'big', items: [EPISODES[0]], next_cursor: 'c2' })
          : res({ path: '/corpus', feed_id: 'big', items: [EPISODES[1]], next_cursor: null })
      }),
    )
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()
    expect(w.find('[data-testid="show-detail-episode-1"]').exists()).toBe(false)

    await w.find('[data-testid="show-detail-load-more"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="show-detail-episode-1"]').text()).toContain('Episode Two')
    expect(w.find('[data-testid="show-detail-load-more"]').exists()).toBe(false)
  })

  it('shows an error state when the episodes request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => res({ detail: 'boom' }, 500)),
    )
    const w = mount(ShowDetailView, { props: { feed: FEED } })
    await flushPromises()

    expect(w.find('[data-testid="show-detail-error"]').exists()).toBe(true)
  })
})
