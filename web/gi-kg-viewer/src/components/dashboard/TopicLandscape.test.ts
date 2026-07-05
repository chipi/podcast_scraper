// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import TopicLandscape from './TopicLandscape.vue'
import { useShellStore } from '../../stores/shell'
import {
  fetchThemeClustersFromApi,
  fetchTopicClustersFromApi,
} from '../../api/corpusTopicClustersApi'

vi.mock('../../api/corpusTopicClustersApi', () => ({
  fetchTopicClustersFromApi: vi.fn(),
  fetchThemeClustersFromApi: vi.fn(),
}))
const fetchClusters = vi.mocked(fetchTopicClustersFromApi)
const fetchThemes = vi.mocked(fetchThemeClustersFromApi)

const DOC = {
  status: 'ok' as const,
  document: {
    clusters: [
      {
        graph_compound_parent_id: 'tc:big',
        canonical_label: 'Big cluster',
        members: [{ topic_id: 'topic:a' }, { topic_id: 'topic:b' }, { topic_id: 'topic:c' }],
      },
      {
        graph_compound_parent_id: 'tc:small',
        canonical_label: 'Small cluster',
        members: [{ topic_id: 'topic:d' }],
      },
    ],
  },
  schemaWarning: null,
}

async function mountIt() {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const w = mount(TopicLandscape)
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return w
}

function diameterOf(el: { attributes: (n: string) => string | undefined }): number {
  const m = /width:\s*(\d+)px/.exec(el.attributes('style') ?? '')
  return m ? Number(m[1]) : 0
}

describe('TopicLandscape', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    fetchClusters.mockResolvedValue(DOC as never)
    fetchThemes.mockResolvedValue(DOC as never)
  })

  it('defaults to the bubbles view with one bubble per cluster', async () => {
    const w = await mountIt()
    expect(w.find('[data-testid="topic-landscape-bubbles"]').exists()).toBe(true)
    expect(w.findAll('[data-testid="topic-landscape-bubbles"] button')).toHaveLength(2)
  })

  it('sizes the bigger cluster larger (area proportional to topic count)', async () => {
    const w = await mountIt()
    const bubbles = w.findAll('[data-testid="topic-landscape-bubbles"] button')
    // Sorted desc → the 3-topic cluster is first and larger than the 1-topic one.
    expect(diameterOf(bubbles[0])).toBeGreaterThan(diameterOf(bubbles[1]))
  })

  it('switches to the compact grid view', async () => {
    const w = await mountIt()
    await w.get('[data-testid="topic-landscape-view-grid"]').trigger('click')
    expect(w.find('[data-testid="topic-landscape-bubbles"]').exists()).toBe(false)
    expect(w.findAll('[aria-label="Topic clusters"] button')).toHaveLength(2)
  })

  it('emits go-graph with the cluster compound id when a bubble is clicked', async () => {
    const w = await mountIt()
    await w.findAll('[data-testid="topic-landscape-bubbles"] button')[0].trigger('click')
    expect(w.emitted('go-graph')?.[0]?.[0]).toBe('tc:big')
  })

  it('source="themes" fetches theme clusters and renders the Theme landscape (#4)', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    const w = mount(TopicLandscape, { props: { source: 'themes' } })
    for (let i = 0; i < 8; i++) await w.vm.$nextTick()
    expect(fetchThemes).toHaveBeenCalled()
    expect(fetchClusters).not.toHaveBeenCalled()
    expect(w.find('[data-testid="intelligence-theme-landscape"]').exists()).toBe(true)
    expect(w.text()).toContain('Theme landscape')
  })
})
