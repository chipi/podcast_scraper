// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import NodeEpisodesSection from './NodeEpisodesSection.vue'
import { fetchNodeEpisodes } from '../../api/corpusLibraryApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

vi.mock('../../api/corpusLibraryApi', () => ({ fetchNodeEpisodes: vi.fn() }))

const ep = (n: number) => ({
  gi_relative_path: '',
  kg_relative_path: `feeds/f/run/ep${n}.kg.json`,
  bridge_relative_path: '',
  episode_id: `ep${n}`,
  episode_title: `Episode ${n}`,
  feed_id: 'f',
  feed_title: 'Show F',
  publish_date: `2026-11-0${n}T10:00:00`,
  metadata_relative_path: `feeds/f/run/ep${n}.metadata.json`,
})

describe('NodeEpisodesSection', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    shell.corpusPath = '/c'
    shell.healthStatus = 'ok'
    vi.mocked(fetchNodeEpisodes).mockReset()
  })

  it('lists the episodes with title, show and date, and opens one in the rail', async () => {
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      path: '/c',
      node_id: 'topic:ai',
      episodes: [ep(2), ep(1)],
      truncated: false,
      total_matched: 2,
    })
    const subject = useSubjectStore()
    const spy = vi.spyOn(subject, 'focusEpisode')
    const w = mount(NodeEpisodesSection, { props: { nodeId: 'k:topic:ai' } })
    await flushPromises()

    expect(fetchNodeEpisodes).toHaveBeenCalledWith('/c', 'k:topic:ai', 10)
    const row = w.get('[data-testid="node-episodes-row-0"]')
    expect(row.text()).toContain('Episode 2')
    expect(row.text()).toContain('Show F')
    expect(row.text()).toContain('2026-11-02')
    expect(w.find('[data-testid="node-episodes-count"]').exists()).toBe(false)
    await row.trigger('click')
    expect(spy).toHaveBeenCalledWith('feeds/f/run/ep2.metadata.json', {
      uiTitle: 'Episode 2',
      episodeId: 'ep2',
    })
  })

  it('says how many there are when it shows only the newest', async () => {
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      path: '/c',
      node_id: 'topic:ai',
      episodes: [ep(3), ep(2)],
      truncated: true,
      total_matched: 37,
    })
    const w = mount(NodeEpisodesSection, { props: { nodeId: 'topic:ai' } })
    await flushPromises()
    expect(w.get('[data-testid="node-episodes-count"]').text()).toBe('(2 newest of 37)')
  })

  it('renders nothing when the node is in no episode', async () => {
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      path: '/c',
      node_id: 'topic:none',
      episodes: [],
      truncated: false,
      total_matched: 0,
    })
    const w = mount(NodeEpisodesSection, { props: { nodeId: 'topic:none' } })
    await flushPromises()
    expect(w.find('[data-testid="node-episodes"]').exists()).toBe(false)
  })
})
