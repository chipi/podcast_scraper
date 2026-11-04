// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { CoverageFeedItem } from '../../api/corpusCoverageApi'
import FeedCoverageTable from './FeedCoverageTable.vue'

const rows: CoverageFeedItem[] = [
  { feed_id: 'f1', display_title: 'Alpha Show', total: 4, with_gi: 1, with_kg: 3 },
  { feed_id: 'f2', display_title: 'Beta Show', total: 3, with_gi: 3, with_kg: 0 },
]

describe('FeedCoverageTable', () => {
  it('shows the empty state when there are no feeds', () => {
    const w = mount(FeedCoverageTable, { props: { rows: [], feedsIndexed: [] } })
    expect(w.text()).toContain('No feeds in catalog.')
    expect(w.find('table').exists()).toBe(false)
    expect(w.text()).not.toContain('lowest GI coverage')
  })

  it('renders a row per feed with rounded GI and KG percentages', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    const trs = w.findAll('[data-testid="feed-coverage-row"]')
    expect(trs).toHaveLength(2)
    expect(trs[0].text()).toContain('Alpha Show')
    expect(trs[0].text()).toContain('25%')
    expect(trs[0].text()).toContain('75%')
    expect(trs[1].text()).toContain('100%')
    expect(trs[1].text()).toContain('0%')
  })

  it('sizes the coverage bars from the percentage', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    expect(w.find('.bg-gi').attributes('style')).toContain('width: 25%')
    expect(w.find('.bg-kg').attributes('style')).toContain('width: 75%')
  })

  it('reports 0% (not NaN) for a feed with no episodes', () => {
    const w = mount(FeedCoverageTable, {
      props: {
        rows: [{ feed_id: 'z', display_title: 'Zero', total: 0, with_gi: 0, with_kg: 0 }],
        feedsIndexed: [],
      },
    })
    expect(w.text()).not.toContain('NaN')
    expect(w.text()).toContain('0%')
    expect(w.text()).not.toContain('lowest GI coverage')
  })

  it('marks indexed feeds Yes using trimmed ids and ignores blank entries', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [' f2 ', '', '  '] } })
    const trs = w.findAll('[data-testid="feed-coverage-row"]')
    expect(trs[0].findAll('td').at(-1)!.text()).toBe('—')
    expect(trs[1].findAll('td').at(-1)!.text()).toBe('Yes')
  })

  it('writes an insight about the first (lowest GI) feed with plural missing count', () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    expect(w.text()).toContain(
      'Feed Alpha Show has lowest GI coverage at 25% — 3 episodes without GI artifacts.',
    )
  })

  it('uses singular wording when one episode lacks GI', () => {
    const w = mount(FeedCoverageTable, {
      props: {
        rows: [{ feed_id: 'f', display_title: 'Solo', total: 2, with_gi: 1, with_kg: 1 }],
        feedsIndexed: [],
      },
    })
    expect(w.text()).toContain('1 episode without GI artifacts.')
  })

  it('emits select-feed with the feed id from every clickable cell', async () => {
    const w = mount(FeedCoverageTable, { props: { rows, feedsIndexed: [] } })
    const cells = w.findAll('[data-testid="feed-coverage-row"]')[1].findAll('td')
    expect(cells).toHaveLength(5)
    for (const c of cells) {
      await c.trigger('click')
    }
    expect(w.emitted('select-feed')).toEqual(Array(5).fill(['f2']))
  })
})
