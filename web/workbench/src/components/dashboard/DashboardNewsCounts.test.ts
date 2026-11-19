// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DashboardNewsCounts from './DashboardNewsCounts.vue'
import { fetchNewsSummary } from '../../api/newsApi'
import { useShellStore } from '../../stores/shell'

vi.mock('../../api/newsApi', () => ({ fetchNewsSummary: vi.fn() }))

describe('DashboardNewsCounts (UXS-019)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    shell.corpusPath = '/c'
    shell.healthStatus = 'ok'
    vi.mocked(fetchNewsSummary).mockReset()
  })

  it('asks nothing without news', async () => {
    mount(DashboardNewsCounts)
    await flushPromises()
    expect(fetchNewsSummary).not.toHaveBeenCalled()
  })

  it('states items, publishers, teasers and paywalled', async () => {
    useShellStore().contentTypes = ['podcast_episode', 'news_article']
    vi.mocked(fetchNewsSummary).mockResolvedValue({
      path: '/c',
      items: 5,
      publishers: 2,
      teasers: 2,
      paywalled: 1,
    })
    const w = mount(DashboardNewsCounts)
    await flushPromises()
    expect(w.get('[data-testid="dashboard-news-counts"]').text()).toBe(
      'News: 5 items from 2 publishers (2 teasers, 1 paywalled)',
    )
  })

  it('leaves out the extras that are zero, and says one item in the singular', async () => {
    useShellStore().contentTypes = ['podcast_episode', 'news_article']
    vi.mocked(fetchNewsSummary).mockResolvedValue({
      path: '/c',
      items: 1,
      publishers: 1,
      teasers: 0,
      paywalled: 0,
    })
    const w = mount(DashboardNewsCounts)
    await flushPromises()
    expect(w.get('[data-testid="dashboard-news-counts"]').text()).toBe('News: 1 item from 1 publisher')
  })
})
