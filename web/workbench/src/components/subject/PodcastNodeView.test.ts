// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import PodcastNodeView from './PodcastNodeView.vue'
import { fetchCorpusEpisodes, fetchCorpusFeeds } from '../../api/corpusLibraryApi'
import { useShellStore } from '../../stores/shell'

vi.mock('../../api/corpusLibraryApi', () => ({
  fetchCorpusFeeds: vi.fn(),
  fetchCorpusEpisodes: vi.fn(),
}))

const feeds = {
  path: '/c',
  feeds: [
    { feed_id: 'nyt-daily', source_id: 'podcast:nyt-daily', display_title: 'The Daily', episode_count: 2 },
    { feed_id: 'other', source_id: 'podcast:other', display_title: 'Planet Money', episode_count: 1 },
  ],
}

describe('PodcastNodeView finds its show', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    shell.corpusPath = '/c'
    shell.healthStatus = 'ok'
    vi.mocked(fetchCorpusFeeds).mockReset().mockResolvedValue(feeds as never)
    vi.mocked(fetchCorpusEpisodes).mockReset().mockResolvedValue({ items: [] } as never)
  })

  it('by KG source id, which is not a slug of the title (GI 4.0 R3b)', async () => {
    // "podcast:nyt-daily" matches no title slug ("thedaily"); the title match alone left it blank.
    mount(PodcastNodeView, { props: { subjectIdOverride: 'k:podcast:nyt-daily' } })
    await flushPromises()
    expect(fetchCorpusEpisodes).toHaveBeenCalledWith('/c', { feedId: 'nyt-daily', limit: 500 })
  })

  it('by title for a title-based id, until R3b', async () => {
    mount(PodcastNodeView, { props: { subjectIdOverride: 'g:podcast:planet-money' } })
    await flushPromises()
    expect(fetchCorpusEpisodes).toHaveBeenCalledWith('/c', { feedId: 'other', limit: 500 })
  })
})
