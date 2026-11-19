// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { CorpusDigestResponse, CorpusDigestTopicBand } from '../../api/digestApi'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import IntelligenceSnapshot from './IntelligenceSnapshot.vue'

function band(id: string, hits: number): CorpusDigestTopicBand {
  return {
    topic_id: id,
    label: `Label ${id}`,
    query: id,
    hits: Array.from({ length: hits }, (_, i) => ({
      metadata_relative_path: null,
      episode_title: `e${i}`,
      feed_id: 'f',
      score: 1,
    })),
  }
}

function digest(rowFeeds: string[], topics: CorpusDigestTopicBand[]): CorpusDigestResponse {
  return {
    path: '/c',
    window: '7d',
    window_start_utc: '',
    window_end_utc: '',
    compact: true,
    rows: rowFeeds.map((feed_id, i) => ({ feed_id, episode_title: `t${i}` })) as never,
    topics,
    topics_unavailable_reason: null,
  }
}

describe('IntelligenceSnapshot', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('shows a loading line until the digest arrives', () => {
    const w = mount(IntelligenceSnapshot, { props: { digest: null } })
    expect(w.text()).toContain('Loading digest…')
    expect(w.text()).not.toContain('Open Digest')
  })

  it('summarises episodes and distinct feeds, pluralised', () => {
    const w = mount(IntelligenceSnapshot, { props: { digest: digest(['a', 'a', 'b'], []) } })
    expect(w.text()).toContain('Last 7 days — 3 new episodes across 2 feeds')
  })

  it('uses singular wording for one episode in one feed', () => {
    const w = mount(IntelligenceSnapshot, { props: { digest: digest(['a'], []) } })
    expect(w.text()).toContain('Last 7 days — 1 new episode across 1 feed')
  })

  it('handles a digest with no rows', () => {
    const d = digest([], [])
    ;(d as { rows: unknown }).rows = undefined
    const w = mount(IntelligenceSnapshot, { props: { digest: d } })
    expect(w.text()).toContain('0 new episodes across 0 feeds')
  })

  it('lists at most the top three topic bands with episode counts', () => {
    const w = mount(IntelligenceSnapshot, {
      props: {
        digest: digest(['a'], [band('t1', 2), band('t2', 1), band('t3', 0), band('t4', 5)]),
      },
    })
    const items = w.findAll('li')
    expect(items).toHaveLength(3)
    expect(items[0].text()).toBe('Label t1 — 2 episodes')
    expect(items[1].text()).toBe('Label t2 — 1 episode')
    expect(items[2].text()).toBe('Label t3 — 0 episodes')
  })

  it('sets a digest handoff and emits open-digest from a topic button', async () => {
    const w = mount(IntelligenceSnapshot, {
      props: { digest: digest(['a'], [band('t1', 2)]) },
    })
    const nav = useDashboardNavStore()
    await w.get('li button').trigger('click')
    expect(nav.pending).toEqual({ kind: 'digest' })
    expect(w.emitted('open-digest')).toHaveLength(1)
  })

  it('sets a digest handoff and emits open-digest from the Open Digest link', async () => {
    const w = mount(IntelligenceSnapshot, { props: { digest: digest([], []) } })
    const nav = useDashboardNavStore()
    const open = w.findAll('button').find((b) => b.text().includes('Open Digest'))!
    await open.trigger('click')
    expect(nav.pending).toEqual({ kind: 'digest' })
    expect(w.emitted('open-digest')).toHaveLength(1)
  })
})
